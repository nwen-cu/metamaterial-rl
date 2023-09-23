def run(data):
    import os
    import sys
    import subprocess
    from pprint import pprint

    import multiprocessing
    import threading
    
    print('Data Received:', str(data))

    print('Cores:', multiprocessing.cpu_count())

    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])

    print('RAM', total_memory)

    print('MP Start Method:', multiprocessing.get_start_method())
    print('Current Process:', multiprocessing.current_process().name)
    print('Parent Process:', multiprocessing.parent_process().name)

    print('Thread Count:', threading.active_count())
    print('Current Thread', threading.current_thread().getName())

    print('Working Dir:', os.getcwd())
    print('Arguments:', sys.argv)

    print('Environment Variables:')
    pprint(dict(os.environ))

    print('Conda Environment:')
    conda_list = subprocess.run('conda info', shell=True, text=True)
    pprint(conda_list.stdout)

    print('Python Modules:')
    pprint(help('modules'))

    return multiprocessing.cpu_count(), total_memory
