from typing import List, Dict
from typing import Any

import os
import sys
import io
import enum
from pathlib import Path
from concurrent.futures import Future
import time
from datetime import datetime
import uuid
import json
import tarfile

import rpyc
from rpyc.core.async_ import AsyncResult
import yaml
from tqdm.auto import tqdm
import xxhash
from lru import LRU
from pymongo import MongoClient
from gridfs import GridFS
import datajson

import paho.mqtt.client as mqtt

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
    
        

    
class TaskSubmissionStatus(enum.Enum):
    SUBMITTED = 0  # A new task submitted to server
    DUPLICATED = 1   # A task shared the same input hash already queued on server
    EXISTED = 2  # A task already existed in the database

        
class SimHubClient:
    def __init__(self, control_node_ip: str, control_node_port: int = 1883,
                 cache_size: int = 10000):
        
        self.session_id = str(uuid.uuid4())
        
        self._closed = False
        
        self.submitted_tasks = list()
        self.result_cache = LRU(cache_size)
        self.cache_loading_count = 0
        
        self.server_state = 'offline'
        
        self.control_node_ip = control_node_ip
        self.control_node_port = control_node_port
        
        self._db = None
        self._dbfs = None
        
        self.mqttc = mqtt.Client()
        # self.mqttc.on_connect = self.on_connect
        # self.mqttc.on_message = self.on_message
        
        
        self.mqttc.connect(control_node_ip, control_node_port)
        
        self.mqttc.subscribe('simhub/server')
        
        
        self.mqttc.subscribe('simhub/database')
        self.mqttc.message_callback_add('simhub/database', self.update_database_connection)
        
        self.mqttc.subscribe(f'simhub/sessions/{self.session_id}')
        self.mqttc.message_callback_add(f'simhub/sessions/{self.session_id}', self.receive_session_msg)
        
        self.mqttc.loop_start()
        
        print('Waiting for database connection..')
        while self._db is None:
            ...
        print('Connected to database')
        
        
    def update_server_state(self, mqttc, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        self.server_state = payload['server_state']
        
    def update_database_connection(self, mqttc, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        self.database_ip = payload['db_host']
        self.database_port = int(payload['db_port'])
        
        self._db = MongoClient(self.database_ip, self.database_port).simdb
        self._dbfs = GridFS(self._db)
        
                
    def receive_session_msg(self, mqttc, userdata, msg):
        ...
        
        
    def close(self):
        self.mqttc.publish(f'simhub/sessions/{self.session_id}', '{"action": "end_session"}')
        self._db.client.close()
        self._closed = True

        
    def set_experiment(self, experiment_file: str):
        experiment_source_dir = Path(experiment_file).parent
        # Load experiment def file
        with open(experiment_file, 'r') as fp:
            exp = yaml.load(fp, Loader=Loader)
        
        self.experiment_name = exp['ExperimentName']
            
        script_tarfo = io.BytesIO()
        script_tarfile = tarfile.open(fileobj=script_tarfo, mode='w')
        for file in exp['ScriptFiles']:
            script_tarfile.add(experiment_source_dir / file, arcname=file)
        script_tarfile.close()
        script_tarfo.seek(0)
        script_file_id = self._dbfs.put(script_tarfo, filename=f'{self.experiment_name}-script-{self.session_id}')
        script_tarfo.close()
        
        data_tarfo = io.BytesIO()
        data_tarfile = tarfile.open(fileobj=data_tarfo, mode='w')
        for file in exp['DataFiles']:
            data_tarfile.add(experiment_source_dir / file, arcname=file)
        data_tarfile.close()
        data_tarfo.seek(0)
        data_file_id = self._dbfs.put(data_tarfo, filename=f'{self.experiment_name}-data-{self.session_id}')
        data_tarfo.close()
        
        self.mqttc.will_set(f'simhub/sessions/{self.session_id}', '{"action": "end_session"}')
        
        self.mqttc.publish('simhub/sessions', json.dumps(
            {
                'session_id': self.session_id, 
                'experiment_config': exp,
            }
        ))


        
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
                if 'remaining_retries' not in task['task']:
                    self._db.tasks.update_one(task, {
                        '$set': {
                            'status': 'ongoing',
                            'session': self.session_id, 
                            'metrics.submit_timestamp': datetime.now(),
                            'task.remaining_retries': 3,
                        },
                        '$unset': {'task.assignment': ''},
                    })
                    self.submitted_tasks.append((input_hash, TaskSubmissionStatus.SUBMITTED))
                    return input_hash, None
                
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
    
    def poll_task_result(self, input_hash: str, no_cache: bool = False) -> Dict[str, Any] | None:
        if not no_cache and input_hash in self.result_cache:
            return self.result_cache[input_hash]
        else:
            task = self._db.tasks.find_one({'experiment': self.experiment_name, 
                                            'input_hash': input_hash})
            if task['status'] == 'done' or task['status'] == 'failed':
                task = unpack_task_data(task)
                # TODO implement input_files and output_files
                if not no_cache: self.result_cache[input_hash] = task
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
        if input_hash in self.result_cache:
            return self.result_cache[input_hash]
        else:
            return self.poll_task_result(input_hash)

    def clear_tasks(self):
        self.submitted_tasks.clear()
        self.cache_loading_count = 0
