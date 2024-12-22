from typing import Iterable, List

from pathlib import Path

from slurm.constants import DEFAULT_JOB_TIME_MINUTES, DEFAULT_MEMORY_GB
from slurm.task.config import SLURMTaskConfig


class Task:
    def __init__(self,
                 task_id: int,
                 config: SLURMTaskConfig):
        self.task_id = task_id
        self.config = config


class TaskCollection:
    def __init__(self,
                 id_: int,
                 tasks: List[Task] = None,
                 script: str = 'run.py',
                 log_dir: str = 'logs',
                 error_dir: str = 'errors',
                 results_dir: str = 'results',
                 device: str = 'cpu',  # or 'gpu'
                 name_prefix: str = '',
                 reserved_time_minutes: int = DEFAULT_JOB_TIME_MINUTES,
                 memory_gb: int = DEFAULT_MEMORY_GB):

        self.id_ = id_  # ID for this task collection
        self.tasks = tasks or []
        self.script = script  # script to run
        self.name_prefix = name_prefix  # for easier job overview in SLURM
        self.device = device
        self.memory_gb = memory_gb
        self.reserved_time_minutes = reserved_time_minutes

        self.results_dir = results_dir
        self.log_dir = log_dir
        self.error_dir = error_dir

        Path(self.log_dir).mkdir(exist_ok=True, parents=True)
        Path(self.error_dir).mkdir(exist_ok=True, parents=True)
        Path(self.results_dir).mkdir(exist_ok=True, parents=True)

    def add_tasks(self, tasks: Iterable[Task]):
        self.tasks.extend(tasks)

    def add_task(self, task: Task):
        self.add_tasks([task])

    def __len__(self):
        return len(self.tasks)

    @property
    def kwargs_string_array(self):
        def make_bash_array_element(task):
            kwargs_string = ' '.join([f'--{key} {value}' for key, value in task.config.script_kwargs.items()])
            return f"'{kwargs_string}'"

        return f'({" ".join(map(make_bash_array_element, self.tasks))})'