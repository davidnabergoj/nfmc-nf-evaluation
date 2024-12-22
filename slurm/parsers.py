from argparse import ArgumentParser

from slurm.constants import INFINITE_WARMUP_ITERATIONS, INFINITE_SAMPLING_ITERATIONS, DEFAULT_PARTITION, \
    LONG_WARMUP_TIME_SECONDS, LONG_SAMPLING_TIME_SECONDS


class SLURMJobArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--results_file', type=str)
        self.add_argument('--memory_gb', type=int)
        self.add_argument('--device', type=str)


class NFMCJobArgumentParser(SLURMJobArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--sampler', type=str)
        self.add_argument('--flow', type=str)
        self.add_argument('--benchmark', type=str)
        self.add_argument('--sampling_time_limit_seconds', type=float)
        self.add_argument('--warmup_time_limit_seconds', type=float)
        self.add_argument('--warmup_iterations', type=int, default=INFINITE_WARMUP_ITERATIONS)
        self.add_argument('--sampling_iterations', type=int, default=INFINITE_SAMPLING_ITERATIONS)


class NFMCArrayJobArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('-s', '--sampling-time-limit-seconds', type=float, default=LONG_SAMPLING_TIME_SECONDS)
        self.add_argument('-w', '--warmup-time-limit-seconds', type=float, default=LONG_WARMUP_TIME_SECONDS)
        self.add_argument('-p', '--partition', type=str, default=DEFAULT_PARTITION)
        self.add_argument('-t', '--tasks-per-collection', type=int, default=256)
        self.add_argument('-c', '--concurrent-tasks', type=int, default=256)
