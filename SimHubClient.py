from typing import List, Dict
from typing import Any

import os
import sys
import enum
from pathlib import Path
from concurrent.futures import Future
import time
from datetime import datetime
import uuid
import json

import rpyc
from rpyc.core.async_ import AsyncResult
import yaml
from tqdm.auto import tqdm
import xxhash
from lru import LRU
from pymongo import MongoClient
from gridfs import GridFS
import datajson

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
    
def hash_file(filename: str | Path) -> str:
    h = xxhash.xxh128()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def hash_dir(path: str | Path = '.', ignore_hidden_files: bool = True) -> Dict[str, str]:
    # Get file list
    path = str(path)
    files = [top + os.sep + f for top, dirs, files in os.walk(path) for f in files]
    if ignore_hidden_files:
        files = [f for f in files 
                 if not any([part.startswith('.') or part.startswith('_') for part in Path(f).parts])]

    return {file.replace(path, '.', 1): hash_file(file) for file in files}

def unpack_task_data(task: Dict[str, Any]) -> Dict[str, Any]:
    if 'input' in task:
        task['input'] = datajson.load_json(task['input'])
    if 'output' in task:
        task['output'] = datajson.load_json(task['output'])
    # TODO implement input_files and output_files
    
    return task
    
    
class RemoteTaskFuture:
    def __init__(self, future_netref, input_hash, client):
        self._future_netref = future_netref
        self._input_hash = input_hash
        self._done = False
        self._result = None
        self._client = client
    
    def done(self) -> bool:
        if self._future_netref:
            if self._future_netref.done():
                self.result()
                return True
            else:
                return False
        else:
            return self._done
                
    def result(self) -> Any:
        if self._future_netref:
            self._result = self._client._db.tasks.find_one({
                'experiment': self._client.experiment_name,
                'input_hash': self._input_hash
            })
            if 'input' in self._result:
                self._result['input'] = datajson.load_json(self._result['input'])
            if 'output' in self._result:
                self._result['output'] = datajson.load_json(self._result['output'])
            if 'output_files' in self._result:
                # TODO Load files
                ...
            self._future_netref = None
            self._done = True
        return self._result
        
        
class RemoteTaskFutureCollection(List[RemoteTaskFuture]):
    def wait(self, pull_interval: float = 5., progress_bar: bool = True):
        if progress_bar:
            pbar = tqdm(total=len(self))
        pending_futures = list(self)
        while len(pending_futures) > 0:
            completed = []
            for f in pending_futures:
                if f.done():
                    completed.append(f)
            for f in completed:
                pending_futures.remove(f)
            if progress_bar:
                pbar.update(len(completed))
            time.sleep(pull_interval)
        if progress_bar:
            pbar.close()
            
    def results(self):
        self.wait(progress_bar=False)
        return [f.result() for f in self]
    
class TaskSubmissionStatus(enum.Enum):
    SUBMITTED = 0  # A new task submitted to server
    DUPLICATED = 1   # A task shared the same input hash already queued on server
    EXISTED = 2  # A task already existed in the database

    
@rpyc.service
class ClientService(rpyc.Service):
    @rpyc.exposed
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        
    @rpyc.exposed
    def debug(self, *args, **kwargs):
        with open('debug.log', 'a') as fp:
            print(*args, **kwargs, file=fp)

        
class SimHubClient:
    def __init__(self, control_node_ip: str, control_node_port: int = 44444, 
                 database_ip: str | None = None, database_port: int = 40000, 
                 cache_size: int = 10000):
        
        self.control_node_ip = control_node_ip
        self.control_node_port = control_node_port
        self.database_ip = database_ip if database_ip else control_node_ip
        self.database_port = database_port
        
        self._conn = rpyc.connect(self.control_node_ip, self.control_node_port, 
                                  service=ClientService, 
                                  config={
                                      'safe_attrs': dict().__dir__() + iter(()).__dir__(),
                                      'sync_request_timeout': None,
                                  })

        self._server = self._conn.root
        
        self._db = MongoClient(self.database_ip, self.database_port).simdb
        self._dbfs = GridFS(self._db)
        
        self.session_id = self._server.session_id()
        
        self.experiment_name = None
        
        self._closed = False
        
        self.submitted_tasks = list()
        self.result_cache = LRU(cache_size)
        self.cache_loading_count = 0

        
    def close(self):
        self._server.close()
        self._conn.close()
        self._db.client.close()
        self._closed = True

        
    def set_experiment(self, experiment_file: str):
        experiment_source_dir = Path(experiment_file).parent
        # Load experiment def file
        with open(experiment_file, 'r') as fp:
            exp = yaml.load(fp, Loader=Loader)
        
        self.experiment_name = exp['ExperimentName']
        
        self._server.set_experiment(exp['ExperimentName'])
        
        script_files = []
        for file in exp['ScriptFiles']:
            path = (experiment_source_dir / file).resolve(strict=True)
            print(path)
            # TODO set permission for server user to read the file
            script_files.append(str(path))
        data_files = []
        for file in exp['DataFiles']:
            path = (experiment_source_dir / file).resolve(strict=True)
            print(path)
            # TODO set permission for server user to read the file
            data_files.append(str(path))
        
        self._server.initialize_experiment_workspace(exp['WorkspaceDir'], script_files, data_files)
        
        self._server.create_parallel_app(exp['Run'], exp.get('TaskResources'))
        
        self._server.initialize_experiment_executors(exp['PreferedNodes'], exp['Modules'], exp['CondaEnvironment'])
        
    def submit_task(self, input_data: Dict[str, Any]):
        """
        Submit a task to remote simulator nodes
        
        Parameters
        ----------
        input_data : Dict[str, Any]
            input data to simulation script, must be serializable by datajson

        Returns
        -------
        string
            Input hash of input_data, generated by datajson
        Dict[str, Any] | None
            If the task is existed, return the task entry from database
            Otherwise, return None

        Raises
        ------
        TODO
        """
        input_json, input_hash = datajson.dump_json(input_data, generate_hash=True)
        
        # TODO maybe add input_files for large input data
        
        # Check local cache
        if input_hash in self.result_cache:
            self.cache_loading_count += 1
            return input_hash, self.result_cache[input_hash]
        
        # Check database
        task = self._db.tasks.find_one({'experiment': self.experiment_name, 
                                        'input_hash': input_hash})
        
        if task:
            if task['status'] == 'ongoing':
                if task['session'] == self.session_id:
                    # Ignore duplicated task
                    self.submitted_tasks.append((input_hash, TaskSubmissionStatus.DUPLICATED))
                    return input_hash, None
                
                else:
                    # Takeover tasks from previous sessions
                    self._db.tasks.update_one(task, {
                        '$set': {
                            'session': self.session_id, 
                            'metrics.submit_timestamp': datetime.now(),
                        },
                        '$unset': {'task.assignment': ''},
                    })
                    self.submitted_tasks.append((input_hash, TaskSubmissionStatus.SUBMITTED))
                    return input_hash, None
                
            elif task['status'] == 'done':
                self.submitted_tasks.append((input_hash, TaskSubmissionStatus.EXISTED))
                task = unpack_task_data(task)
                return input_hash, task
            
            elif task['status'] == 'failed':
                if task['task']['remaining_retries'] > 0:
                    self._db.tasks.update_one(task, {
                        '$set': {
                            'status': 'ongoing',
                            'session': self.session_id, 
                            'metrics.submit_timestamp': datetime.now(),
                        },
                        '$unset': {'task.assignment': ''},
                        '$inc': {'task.remaining_retries': -1},
                    })
                    self.submitted_tasks.append((input_hash, TaskSubmissionStatus.SUBMITTED))
                    return input_hash, None
                
                else:
                    task = unpack_task_data(task)
                    return input_hash, task
                
                
                
        # Submit new task
        self._db.tasks.insert_one({'experiment': self.experiment_name, 
                                   'session': self.session_id, 
                                   'input': input_json,
                                   'input_hash': input_hash, 
                                   'status': 'ongoing',
                                   'metrics': {'submit_timestamp': datetime.now()}})
        
        self.submitted_tasks.append((input_hash, TaskSubmissionStatus.SUBMITTED))
        return input_hash, None
    
    def poll_task_result(self, input_hash: str) -> Dict[str, Any] | None:
        if input_hash in self.result_cache:
            return self.result_cache[input_hash]
        else:
            task = self._db.tasks.find_one({'experiment': self.experiment_name, 
                                            'input_hash': input_hash})
            if task['status'] == 'done' or task['status'] == 'failed':
                task = unpack_task_data(task)
                # TODO implement input_files and output_files
                self.result_cache[input_hash] = task
                return task
            else:
                return None
    
    def wait(self, interval: float = 10, print_stats: bool = True, progress_bar: bool = True):
        if len(self.submitted_tasks) == 0: return
        existed = 0
        duplicated = 0
        submitted = 0
        for task in self.submitted_tasks:
            if task[1] == TaskSubmissionStatus.EXISTED:
                existed += 1
            elif task[1] == TaskSubmissionStatus.DUPLICATED:
                duplicated += 1
            elif task[1] == TaskSubmissionStatus.SUBMITTED:
                submitted += 1
        if print_stats:
            total_tasks = len(self.submitted_tasks) + self.cache_loading_count
            print(f'Cached: {self.cache_loading_count}({self.cache_loading_count / total_tasks * 100:.2f}%)')
            print(f'Loaded: {existed}({existed / total_tasks * 100:.2f}%)')
            print(f'Duplicated: {duplicated}({duplicated / total_tasks * 100:.2f}%)')
            print(f'New: {submitted}({submitted / total_tasks * 100:.2f}%)')
        if progress_bar:
            pbar = tqdm(total=len(self.submitted_tasks))
        results = [None] * len(self.submitted_tasks)
        # Loop/wait until all result done
        while sum(result is None for result in results) > 0:
            updated = 0
            # Iter over each task
            for i in range(len(self.submitted_tasks)):
                if results[i]:
                    continue
                time.sleep(min(interval / len(self.submitted_tasks), 0.1))
                input_hash, status = self.submitted_tasks[i]
                task_result = self.poll_task_result(input_hash)
                if task_result:
                    results[i] = task_result
                if progress_bar and task_result:
                    pbar.update(1)
                    updated += 1
            # if progress_bar:
            #     pbar.update(updated)
            # time.sleep(interval)
        if progress_bar:
            pbar.close()
            
        successful = []
        failed = []
        for result in results:
            if result['status'] == 'done':
                successful.append(result)
            elif result['status'] == 'failed':
                failed.append(result)
        if print_stats:
            total_tasks = len(self.submitted_tasks) + self.cache_loading_count
            print(f'Successful: {len(successful)}({len(successful) / total_tasks * 100:.2f}%)')
            print(f'Failed: {len(failed)}({len(failed) / total_tasks * 100:.2f}%)')
            
    def wait_for_task(self, input_hash: str) -> Dict[str, Any]:
        while not (task_result:= self.poll_task_result(input_hash)):
            time.sleep(0.5)
        return task_result
            
    
    def get_result(self, input_hash: str) -> Dict[str, Any]:
        return self.result_cache[input_hash]

    def clear_tasks(self):
        self.submitted_tasks.clear()
        self.cache_loading_count = 0
