from slurm.constants import LONG_WARMUP_TIME_SECONDS, LONG_SAMPLING_TIME_SECONDS
from slurm.util import NFMCStrategy


class SLURMTaskConfig:
    def __init__(self,
                 **kwargs):
        self.script_kwargs = kwargs

    def __repr__(self):
        return str(self.script_kwargs)


class NFMCTaskConfig(SLURMTaskConfig):
    def __init__(self,
                 strategy: NFMCStrategy,
                 benchmark: str,
                 warmup_time_limit_seconds: float = LONG_WARMUP_TIME_SECONDS,
                 sampling_time_limit_seconds: float = LONG_SAMPLING_TIME_SECONDS,
                 **kwargs):
        super().__init__(**kwargs)
        self.script_kwargs.update(
            sampler=strategy.sampler,
            flow=strategy.flow,
            benchmark=benchmark,
            warmup_time_limit_seconds=warmup_time_limit_seconds,
            sampling_time_limit_seconds=sampling_time_limit_seconds,
        )
