import itertools

from typing import Iterable, Dict, Type

from slurm.task.config import SLURMTaskConfig, NFMCTaskConfig


def make_configs_product(variables: Dict[str, Iterable],
                         config_class: Type[SLURMTaskConfig] = NFMCTaskConfig,
                         **constant_kwargs):
    configs = []
    for config_values in itertools.product(*variables.values()):
        config_kwargs = constant_kwargs
        config_kwargs.update({key: value for key, value in zip(variables.keys(), config_values)})
        config = config_class(**config_kwargs)
        configs.append(config)
    return configs
