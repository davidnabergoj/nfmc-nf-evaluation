import pathlib
from dataclasses import dataclass
from typing import Optional, Union, Iterable, Tuple, List, Any, Dict
import torch

from nfmc import sample
from nfmc.util import parse_flow_string
from potentials.base import Potential
import json


@dataclass
class BenchmarkOutput:
    benchmark: str
    sampler: str
    flow: str
    estimated_first_moment: torch.Tensor
    estimated_second_moment: torch.Tensor
    true_first_moment: torch.Tensor = None
    true_second_moment: torch.Tensor = None
    statistics: Optional[dict] = None
    n_flow_params: Optional[int] = None

    def __post_init__(self):
        self.statistics = self.statistics or {}

    @property
    def first_moment_abs_error(self) -> float:
        if self.true_first_moment is None:
            return torch.nan
        return float(torch.max(torch.abs(self.estimated_first_moment - self.true_first_moment)))

    @property
    def second_moment_abs_error(self) -> float:
        if self.true_second_moment is None:
            return torch.nan
        return float(torch.max(torch.abs(self.estimated_second_moment - self.true_second_moment)))

    @property
    def second_moment_rel_error(self) -> float:
        if self.true_second_moment is None:
            return torch.nan
        return float(
            torch.max(torch.abs(self.estimated_second_moment - self.true_second_moment) / self.true_second_moment)
        )

    @property
    def first_moment_squared_bias(self) -> float:
        if self.true_first_moment is None or self.true_second_moment is None:
            return torch.nan
        true_variance = self.true_second_moment - self.true_first_moment ** 2
        return float(torch.max(torch.abs(self.estimated_first_moment - self.true_first_moment) ** 2 / true_variance))

    @property
    def second_moment_squared_bias(self) -> float:
        if self.true_first_moment is None or self.true_second_moment is None:
            return torch.nan
        true_variance = self.true_second_moment - self.true_first_moment ** 2
        return float(torch.max(torch.abs(self.estimated_second_moment - self.true_second_moment) ** 2 / true_variance))

    def __dict__(self):
        return {
            'sampler': self.sampler,
            'flow': self.flow,
            'benchmark': self.benchmark,
            'first_moment_abs_error': float(self.first_moment_abs_error),
            'second_moment_abs_error': float(self.second_moment_abs_error),
            'second_moment_rel_error': float(self.second_moment_rel_error),
            'first_moment_squared_bias': float(self.first_moment_squared_bias),
            'second_moment_squared_bias': float(self.second_moment_squared_bias),
            'n_flow_params': int(self.n_flow_params),
            **self.statistics
        }

    def save(self, file: pathlib.Path, **kwargs):
        output_data = self.__dict__()
        output_data.update(kwargs)
        for k, v in output_data.items():
            if isinstance(v, torch.Tensor):
                print(k)
                print(v)
                raise ValueError
        with open(file, 'w') as f:
            json.dump(output_data, f)


class Benchmark:
    def __init__(self,
                 strategies: Iterable[Union[Tuple[str, Optional[str]], str]],
                 # strategies[i] = sampler_i, flow_i or just sampler_i
                 target: Potential,
                 n_chains: int = 100,
                 x0: torch.Tensor = None,
                 output_dir: Optional[pathlib.Path] = None,
                 n_sampling_iterations: int = 5000,
                 n_warmup_iterations: int = 1000,
                 sampling_time_limit_seconds: float = None,
                 warmup_time_limit_seconds: float = None,
                 output_files: Optional[List[str]] = None,
                 name: str = 'undefined',
                 extra_data: Optional[Dict] = None,
                 **kwargs):
        """

        :param strategies: iterable of sampling strategies. Each strategy is specified as either a string denoting a regular sampler or a tuple of (sampler, flow), denoting the NFMC sampler and normalizing flow.
        :param torch.Tensor x0: initial states.
        :param Potential target: target potential.
        :param n_chains:
        :param output_dir:
        :param n_sampling_iterations:
        :param n_warmup_iterations:
        :param sampling_time_limit_seconds:
        :param warmup_time_limit_seconds:
        :param output_files:
        :param name:
        :param kwargs:
        """
        self.name = name
        self.sampling_strategies = [(s, None) if isinstance(s, str) else s for s in strategies]
        self.target = target

        if x0 is None:
            self.x0 = torch.randn(size=(n_chains, *target.event_shape)) * 2
        else:
            self.x0 = x0
        self.output_dir = output_dir
        self.n_chains = len(self.x0)
        self.n_sampling_iterations = n_sampling_iterations
        self.n_warmup_iterations = n_warmup_iterations
        self.sample_kwargs = kwargs
        self.sampling_time_limit_seconds = sampling_time_limit_seconds
        self.warmup_time_limit_seconds = warmup_time_limit_seconds
        self.output_files = output_files
        self.extra_data = extra_data or {}
        self.extra_data['warmup_time_limit_seconds'] = warmup_time_limit_seconds
        self.extra_data['sampling_time_limit_seconds'] = sampling_time_limit_seconds

    def run_single(self, strategy: Tuple[str, Optional[str]], device: str = None) -> BenchmarkOutput:
        torch.random.fork_rng()
        torch.manual_seed(0)
        sampler, flow = strategy

        output = sample(
            target=self.target,
            strategy=sampler,
            flow=flow,
            x0=self.x0,
            **{
                **dict(
                    warmup=True,
                    n_iterations=self.n_sampling_iterations,
                    n_warmup_iterations=self.n_warmup_iterations,
                    n_chains=self.n_chains,
                    sampling_time_limit_seconds=self.sampling_time_limit_seconds,
                    warmup_time_limit_seconds=self.warmup_time_limit_seconds,
                    show_progress=False,
                    param_kwargs={'store_samples': False},
                    device=device
                ),
                **self.sample_kwargs,
            }
        )
        if str(flow) == 'None':
            n_flow_params = 0
        else:
            n_flow_params = sum([int(p.numel()) for p in output.kernel.flow.parameters()])

        return BenchmarkOutput(
            benchmark=self.name,
            flow=flow,
            sampler=sampler,
            estimated_first_moment=output.mean,
            estimated_second_moment=output.second_moment,
            true_first_moment=self.target.mean,
            true_second_moment=self.target.second_moment,
            n_flow_params=n_flow_params,
            statistics=output.statistics.__dict__(),
        )

    def run(self, **kwargs) -> list[tuple[Any, BenchmarkOutput]]:
        run_outputs = []
        for i, strategy in enumerate(self.sampling_strategies):
            print(f"Running: {strategy}")
            try:
                output: BenchmarkOutput = self.run_single(strategy, **kwargs)
            except ValueError as e:
                print(f"Error running {strategy}")
                print(e)
                continue

            sampler = strategy[0]
            flow_data = parse_flow_string(strategy[1])

            output_file = None
            if self.output_files is not None:
                output_file = self.output_files[i]
            elif self.output_dir is not None:
                if not self.output_dir.exists():
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                if self.output_dir.exists() and self.output_dir.is_dir():
                    output_file = self.output_dir / f"{sampler}-{flow_data['name']}-{flow_data['hash']}.json"

            if output_file is not None:
                output.save(
                    output_file,
                    flow_name=flow_data['name'],  # string
                    flow_kwargs=flow_data['kwargs'],  # dictionary
                    flow=flow_data['name'],  # string
                    benchmark=self.name,  # string
                    sampler=sampler,  # string
                    **self.extra_data  # dictionary
                )
            run_outputs.append((*strategy, output))
        return run_outputs

    def __repr__(self):
        return f'{self.__class__.__name__} [{self.n_chains} chains, {self.n_sampling_iterations} steps, {self.sampling_strategies}]'
