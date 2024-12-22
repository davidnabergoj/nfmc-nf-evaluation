import pathlib

import torch

from benchmarking.base import Benchmark
from potentials.real.basketball import BasketballV1, BasketballV2
from potentials.real.biochemical_oxygen_demand import BiochemicalOxygenDemand
from potentials.real.eight_schools import EightSchools
from potentials.real.german_credit import GermanCredit, SparseGermanCredit
from potentials.real.item_response import SyntheticItemResponseTheory
from potentials.real.radon import RadonVaryingSlopes, RadonVaryingIntercepts, RadonVaryingInterceptsAndSlopes
from potentials.real.stochastic_volatility import StochasticVolatilityModel
from potentials.synthetic.funnel import Funnel
from potentials.synthetic.phi4 import Phi4
from potentials.synthetic.rosenbrock import Rosenbrock

__all__ = [
    'RosenbrockBenchmark',
    'BiochemicalOxygenDemandBenchmark',
    'SparseGermanCreditBenchmark',
    'GermanCreditBenchmark',
    'BasketballV1Benchmark',
    'BasketballV2Benchmark',
    'FunnelBenchmark',
    'RadonVaryingInterceptsAndSlopesBenchmark',
    'RadonVaryingInterceptsBenchmark',
    'RadonVaryingSlopesBenchmark',
    'EightSchoolsBenchmark',
    'SyntheticItemResponseTheoryBenchmark',
    'StochasticVolatilityModelBenchmark',
    'Phi4Side8',
    'Phi4Side16',
    'Phi4Side32',
    'Phi4Side64',
    'Phi4Side128',
    'Phi4Side256'
]


class FunnelBenchmark(Benchmark):
    def __init__(self, event_shape=(100,), scale: float = 3.0, **kwargs):
        target = Funnel(event_shape=event_shape, scale=scale)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class RosenbrockBenchmark(Benchmark):
    def __init__(self, event_shape=(100,), scale: float = 10.0, **kwargs):
        target = Rosenbrock(event_shape=event_shape, scale=scale)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class Phi4Side8(Benchmark):
    def __init__(self, **kwargs):
        target = Phi4(length=8, add_channel_dimension=True)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class Phi4Side16(Benchmark):
    def __init__(self, **kwargs):
        target = Phi4(length=16, add_channel_dimension=True)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class Phi4Side32(Benchmark):
    def __init__(self, **kwargs):
        target = Phi4(length=32, add_channel_dimension=True)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class Phi4Side64(Benchmark):
    def __init__(self, **kwargs):
        target = Phi4(length=64, add_channel_dimension=True)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class Phi4Side128(Benchmark):
    def __init__(self, **kwargs):
        target = Phi4(length=128, add_channel_dimension=True)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class Phi4Side256(Benchmark):
    def __init__(self, **kwargs):
        target = Phi4(length=256, add_channel_dimension=True)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class RadonVaryingSlopesBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = RadonVaryingSlopes()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class RadonVaryingInterceptsBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = RadonVaryingIntercepts()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class RadonVaryingInterceptsAndSlopesBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = RadonVaryingInterceptsAndSlopes()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class GermanCreditBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = GermanCredit()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class SparseGermanCreditBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = SparseGermanCredit()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class BiochemicalOxygenDemandBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = BiochemicalOxygenDemand()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class BasketballV1Benchmark(Benchmark):
    def __init__(self, file_path, **kwargs):
        target = BasketballV1(file_path)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class BasketballV2Benchmark(Benchmark):
    def __init__(self, file_path, **kwargs):
        target = BasketballV2(file_path)
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class EightSchoolsBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = EightSchools()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class SyntheticItemResponseTheoryBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = SyntheticItemResponseTheory()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)


class StochasticVolatilityModelBenchmark(Benchmark):
    def __init__(self, **kwargs):
        target = StochasticVolatilityModel()
        torch.random.fork_rng()
        torch.manual_seed(0)
        super().__init__(target=target, **kwargs)
        self.x0 = 1.5 * torch.randn_like(self.x0) / 10


if __name__ == '__main__':
    strategies = [
        'hmc',
        ('imh', 'glow'),
        ('jump_hmc', 'ms-realnvp'),
        ('neutra_hmc', 'ms-nice'),
    ]

    Phi4Side8(
        strategies=strategies,
        output_dir=pathlib.Path("output/phi4_side8"),
        n_iterations=3,
        warmup=False,
    ).run()
