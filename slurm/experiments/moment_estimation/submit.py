from argparse import ArgumentParser

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from slurm.util import print_submit_countdown
from slurm.client import SLURMClient
from slurm.job import SLURMJobArray
from slurm.constants import TARGETS_REAL_HIERARCHICAL, TARGETS_SYNTHETIC_MULTIMODAL, TARGETS_IMAGE, DEFAULT_PARTITION

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--partition', type=str, default=DEFAULT_PARTITION)
    args = parser.parse_args()

    warmup_time_seconds = 6 * 3600
    sampling_time_seconds = 36 * 3600
    total_time_seconds = warmup_time_seconds + sampling_time_seconds
    total_time_minutes = total_time_seconds // 60

    client = SLURMClient()
    counter = 0
    jobs = []
    for target in TARGETS_REAL_HIERARCHICAL + TARGETS_SYNTHETIC_MULTIMODAL + ['rosenbrock'] + TARGETS_IMAGE:
        job = SLURMJobArray(
            id_=counter,
            reserved_time_minutes=total_time_minutes + 10,
            name_prefix=f'mmnt[{counter}; {target}]-',
            script_kwargs={
                'sampling_time_limit_seconds': sampling_time_seconds,
                'warmup_time_limit_seconds': warmup_time_seconds,
            }
        )
        jobs.append(job)
        counter += 1

    print(f'Total jobs: {len(jobs)}')
    print_submit_countdown()
    client.submit_jobs(jobs, partition=args.partition)