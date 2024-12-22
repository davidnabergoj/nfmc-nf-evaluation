from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[3]))

from slurm.task.config import NFMCTaskConfig
from slurm.experiments.util import make_configs_product
from slurm.job import SLURMJobArray
from slurm.parsers import NFMCArrayJobArgumentParser
from slurm.client import SLURMClient
from slurm.constants import ROOT_DIR, MEDIUM_SAMPLING_TIME_SECONDS, MEDIUM_WARMUP_TIME_SECONDS, \
    DEFAULT_OVERHEAD_TIME_MINUTES
from slurm.util import print_submit_countdown, get_strategies_single_config

if __name__ == '__main__':
    parser = NFMCArrayJobArgumentParser()
    args = parser.parse_args()
    args.sampling_time_limit_seconds = MEDIUM_SAMPLING_TIME_SECONDS
    args.warmup_time_limit_seconds = MEDIUM_WARMUP_TIME_SECONDS

    # Main variables
    device = 'cpu'
    reserved_time_minutes = int(
        args.sampling_time_limit_seconds // 60 +
        args.warmup_time_limit_seconds // 60 +
        DEFAULT_OVERHEAD_TIME_MINUTES
    )
    name_prefix = 'dim-'
    experiment_dir = f'{ROOT_DIR}/slurm/experiments/dimensionality_ablation'
    script = f'{experiment_dir}/run.py'

    submission_log_file = f'{experiment_dir}/submit.log'
    results_dir = f'{experiment_dir}/results'
    log_dir = f'{experiment_dir}/logs'
    error_dir = f'{experiment_dir}/errors'

    configs = make_configs_product(
        {
            'dimensionality': [2, 10, 100, 1000, 10_000],
            'strategy': get_strategies_single_config(),
            'benchmark': ['diagonal_gaussian'],
            'device': [device],
        },
        sampling_time_limit_seconds=args.sampling_time_limit_seconds,
        warmup_time_limit_seconds=args.warmup_time_limit_seconds,
    )

    client = SLURMClient()
    job_array = SLURMJobArray(
        task_configs=configs,
        tasks_per_collection=args.tasks_per_collection,
        reserved_time_minutes=reserved_time_minutes,
        name_prefix=name_prefix,
        script=script,
        result_dir=results_dir,
        log_dir=log_dir,
        error_dir=error_dir,
        memory_gb=128
    )

    print(job_array)
    print_submit_countdown()
    client.submit_jobs(
        job_array.task_collections,
        concurrent_tasks=args.concurrent_tasks,
        submission_log_file=submission_log_file
    )
