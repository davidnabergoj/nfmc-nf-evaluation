import pathlib

import torch

from benchmarking.base import Benchmark
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian1
from potentials.synthetic.gaussian.full_rank import FullRankGaussian1, FullRankGaussian0
from potentials.synthetic.gaussian.unit import StandardGaussian

__all__ = [
    'StandardGaussianBenchmark',
    'DiagonalGaussianBenchmark',
    'FullRankGaussianBenchmark',
    'IllConditionedFullRankGaussianBenchmark',
]


class StandardGaussianBenchmark(Benchmark):
    def __init__(self, event_shape=(100,), **kwargs):
        n_chains = 100
        target = StandardGaussian(event_shape=event_shape)
        x0 = torch.randn(size=(n_chains, *target.event_shape)) * 2
        super().__init__(x0=x0, target=target, **kwargs)


class DiagonalGaussianBenchmark(Benchmark):
    def __init__(self, event_shape=(100,), **kwargs):
        n_chains = 100
        target = DiagonalGaussian1(event_shape=event_shape)
        x0 = torch.randn(size=(n_chains, *target.event_shape)) * 2
        super().__init__(x0=x0, target=target, **kwargs)


class FullRankGaussianBenchmark(Benchmark):
    def __init__(self, event_shape=(100,), **kwargs):
        n_chains = 100
        target = FullRankGaussian1(n_dim=event_shape[0])
        x0 = torch.randn(size=(n_chains, *target.event_shape)) * 2
        super().__init__(x0=x0, target=target, **kwargs)


class IllConditionedFullRankGaussianBenchmark(Benchmark):
    def __init__(self, event_shape=(100,), **kwargs):
        n_chains = 100
        target = FullRankGaussian0(n_dim=event_shape[0])
        x0 = torch.randn(size=(n_chains, *target.event_shape)) * 2
        super().__init__(x0=x0, target=target, **kwargs)


if __name__ == '__main__':
    strategies = [
        ('jump_hmc', 'realnvp'),
        ('neutra_hmc', 'realnvp'),
        ('imh', 'realnvp'),
        'hmc',
    ]

    StandardGaussianBenchmark(
        strategies=strategies,
        output_dir=pathlib.Path("output/standard_gaussian")
    ).run()

    DiagonalGaussianBenchmark(
        strategies=strategies,
        output_dir=pathlib.Path("output/diagonal_gaussian")
    ).run()

    FullRankGaussianBenchmark(
        strategies=strategies,
        output_dir=pathlib.Path("output/full_rank_gaussian")
    ).run()
