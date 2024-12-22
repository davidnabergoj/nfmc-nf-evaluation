from typing import List
from pathlib import Path
from slurm.constants import DEFAULT_MEMORY_GB, DEFAULT_JOB_TIME_MINUTES
from slurm.task.base import TaskCollection, Task
from slurm.task.config import SLURMTaskConfig


class SLURMJobArray:
    def __init__(self,
                 task_configs: List[SLURMTaskConfig],
                 script: str = 'run.py',
                 log_dir: str = 'logs',
                 error_dir: str = 'errors',
                 result_dir: str = 'results',
                 reserved_time_minutes: int = DEFAULT_JOB_TIME_MINUTES,
                 memory_gb: int = DEFAULT_MEMORY_GB,
                 name_prefix: str = '',
                 tasks_per_collection: int = 100,
                 **kwargs):
        """

        :param kwargs: keyword arguments to be passed to SLURMJob.
        """
        super().__init__()
        self.job_kwargs = kwargs
        self.job_kwargs.update(
            dict(
                reserved_time_minutes=reserved_time_minutes,
                memory_gb=memory_gb,
                name_prefix=name_prefix,
                script=script,
                results_dir=result_dir,
                log_dir=log_dir,
                error_dir=error_dir,
            )
        )

        # Create directories for results, logs, and errors
        Path(result_dir).mkdir(exist_ok=True, parents=True)
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        Path(error_dir).mkdir(exist_ok=True, parents=True)

        drop_mask = []
        for i in range(len(task_configs)):
            task_configs[i].script_kwargs.update(
                results_file=str(Path(result_dir) / f'{i}.json')
            )
            drop_this_task = False
            if Path(task_configs[i].script_kwargs['results_file']).exists():
                if 'benchmark' in task_configs[i].script_kwargs:
                    if not ('full_rank' in task_configs[i].script_kwargs['benchmark']):
                        drop_this_task = True
            drop_mask.append(drop_this_task)
        task_configs = [t for t, d in zip(task_configs, drop_mask) if not d]

        # split task configurations into those with cpus and those with gpus
        cpu_configs = []
        gpu_configs = []
        for c in task_configs:
            if 'device' in c.script_kwargs:
                if c.script_kwargs['device'] == 'gpu':
                    gpu_configs.append(c)
                else:
                    cpu_configs.append(c)
            else:
                cpu_configs.append(c)

        cpu_task_collections = []
        for job_id, job_task_configs in enumerate(self._divide_into_chunks(list(cpu_configs), tasks_per_collection)):
            cpu_task_collections.append(
                TaskCollection(
                    id_=job_id,
                    tasks=[Task(i, c) for i, c in enumerate(job_task_configs)],
                    device='',
                    **self.job_kwargs
                )
            )

        gpu_task_collections = []
        for job_id, job_task_configs in enumerate(self._divide_into_chunks(list(gpu_configs), tasks_per_collection)):
            gpu_task_collections.append(
                TaskCollection(
                    id_=job_id,
                    tasks=[Task(i, c) for i, c in enumerate(job_task_configs)],
                    device='gpu',
                    **self.job_kwargs
                )
            )

        self.cpu_task_collections = cpu_task_collections
        self.gpu_task_collections = gpu_task_collections
        self.task_collections = gpu_task_collections + cpu_task_collections

        for c in self.task_collections:
            assert len(c) <= tasks_per_collection, \
                f"Expected each task collection to have at most {tasks_per_collection} tasks, but got {len(c)}"

    @staticmethod
    def _divide_into_chunks(full_list: List, chunk_size: int = 100):
        for i in range(0, len(full_list), chunk_size):
            yield full_list[i:i + chunk_size]

    def __repr__(self):
        if len(self.task_collections) == 1:
            task_count_string = f'{len(self.task_collections[0])}'
        elif len(self.task_collections[0]) == len(self.task_collections[-1]):
            task_count_string = f'{len(self.task_collections)} x {len(self.task_collections[0])}'
        else:
            task_count_string = f'{len(self.task_collections) - 1} x {len(self.task_collections[0])} + {len(self.task_collections[-1])}'

        return f'Job array with {task_count_string} tasks'
