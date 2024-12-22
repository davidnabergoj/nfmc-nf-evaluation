from typing import Optional, Iterable

import subprocess
from pathlib import Path
import time

from slurm.constants import DEFAULT_PYTHONPATH, DEFAULT_PYTHON_EXECUTABLE, DEFAULT_GPU_PARTITION, DEFAULT_PARTITION
from slurm.job import SLURMJobArray
from slurm.task.base import TaskCollection


class SLURMClient:
    def __init__(self,
                 python_executable: str = DEFAULT_PYTHON_EXECUTABLE,
                 pythonpath: str = DEFAULT_PYTHONPATH):
        self.python_executable = python_executable
        self.pythonpath = pythonpath

    def submit_job_array(self, job_array: SLURMJobArray, **kwargs):
        for collection in job_array.task_collections:
            self.submit_job(collection, **kwargs)

    def submit_jobs(self, jobs: Iterable[TaskCollection], submission_log_file: str = None, **kwargs):
        total_tasks = 0
        for job in jobs:
            self.submit_job(job, **kwargs)
            total_tasks += len(job.tasks)

        if submission_log_file:
            with open(submission_log_file, 'w') as f:
                f.write(str(total_tasks))

    def submit_job(self,
                   task_collection: TaskCollection,
                   concurrent_tasks: int):
        python_call = ' '.join([self.python_executable, task_collection.script])
        sbatch_script_lines = [
            f"#!/bin/bash",
            f'#SBATCH --array=0-{len(task_collection) - 1}%{concurrent_tasks}',
            f'#SBATCH --ntasks=1',
            f"#SBATCH --time={task_collection.reserved_time_minutes}",
            f"#SBATCH --mem={task_collection.memory_gb}G",
            f"#SBATCH --output={task_collection.log_dir}/{task_collection.id_}-%a.log",
            f"#SBATCH --error={task_collection.error_dir}/{task_collection.id_}-%a.log",
            f"#SBATCH --job-name={task_collection.name_prefix}{task_collection.id_}-{abs(hash(task_collection))}",
            f"{'#SBATCH --gres=gpu:1' if task_collection.device == 'gpu' else ''}",
            f"{f'#SBATCH --partition={DEFAULT_GPU_PARTITION}' if task_collection.device == 'gpu' else ''}",
            self.pythonpath,
            f'kwargs_string_array={task_collection.kwargs_string_array}',
            f'kwargs_string=${{kwargs_string_array[SLURM_ARRAY_TASK_ID]}}',
            'srun ' + python_call + f' ${{kwargs_string}}',
        ]

        filename = str(abs(hash(time.time()))) + '.txt'
        with open(filename, "w") as f:
            f.writelines([line + "\n" for line in sbatch_script_lines])
        subprocess.run(["sbatch", filename])
        Path(filename).unlink()
