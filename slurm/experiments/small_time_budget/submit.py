from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[3]))


from slurm.experiments.util import make_configs_product
from slurm.job import SLURMJobArray
from slurm.parsers import NFMCArrayJobArgumentParser
from slurm.client import SLURMClient
from slurm.constants import ROOT_DIR, TARGETS_NON_IMAGE, DEFAULT_OVERHEAD_TIME_MINUTES
from slurm.util import print_submit_countdown, get_strategies_single_config

if __name__ == '__main__':
    parser = NFMCArrayJobArgumentParser()
    args = parser.parse_args()

    print("Setting warmup time to 2 minutes")
    warmup_time_limit_seconds = 2 * 60

    print("Setting sampling time to 5 minutes")
    sampling_time_limit_seconds = 5 * 60


    # Main variables
    device = 'cpu'
    reserved_time_minutes = int(sampling_time_limit_seconds // 60 + warmup_time_limit_seconds // 60 + DEFAULT_OVERHEAD_TIME_MINUTES)
    name_prefix = 'stb-'
    experiment_dir = f'{ROOT_DIR}/slurm/experiments/small_time_budget'
    script = f'{ROOT_DIR}/slurm/experiments/general_experiment/run.py'

    submission_log_file = f'{experiment_dir}/submit.log'
    results_dir = f'{experiment_dir}/results'
    log_dir = f'{experiment_dir}/logs'
    error_dir = f'{experiment_dir}/errors'

    configs = make_configs_product(
        {
            'strategy': get_strategies_single_config(),
            'benchmark': TARGETS_NON_IMAGE,
            'device': [device],
        },
        sampling_time_limit_seconds=sampling_time_limit_seconds,
        warmup_time_limit_seconds=warmup_time_limit_seconds,
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
    )

    print(job_array)
    print_submit_countdown()
    client.submit_jobs(
        job_array.task_collections,
        concurrent_tasks=args.concurrent_tasks,
        submission_log_file=submission_log_file
    )
