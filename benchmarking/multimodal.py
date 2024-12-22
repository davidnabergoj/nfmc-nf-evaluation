import pathlib
from typing import Tuple

import torch

from benchmarking.base import Benchmark
from potentials.synthetic.double_well import DoubleWell
from potentials.synthetic.multimodal import TripleGaussian3, RandomlyPositionedGaussians, BoundedGaussianChain
from potentials.synthetic.shell import DoubleGammaShell

__all__ = [
    'SeparatedMultimodalBenchmark',
    'OverlappingMultimodalBenchmark',
    'DoubleShellBenchmark',
    'SmallDoubleWellBenchmark',
    'BigDoubleWellBenchmark'
]


class SeparatedMultimodalBenchmark(Benchmark):
    # TODO Call this 'evenly spaced'
    def __init__(self,
                 event_shape: Tuple[int] = (100,),
                 n_components: int = 3,
                 weight_scale: float = 0.0,
                 **kwargs):
        torch.random.fork_rng()
        torch.manual_seed(0)
        n_chains = 100
        if n_components == 3:
            target = TripleGaussian3(event_shape)
        else:
            target = BoundedGaussianChain(event_shape, n_components=n_components, weight_scale=weight_scale)

        super().__init__(
            x0=torch.randn(size=(n_chains, *target.event_shape)) * 2,
            target=target,
            **kwargs
        )


class OverlappingMultimodalBenchmark(Benchmark):
    # TODO Call this randomly positioned
    def __init__(self,
                 event_shape: Tuple[int] = (100,),
                 n_components: int = 20,
                 weight_scale: float = 1.0,
                 **kwargs):
        torch.random.fork_rng()
        torch.manual_seed(0)
        n_chains = 100
        target = RandomlyPositionedGaussians(
            event_shape=event_shape,
            n_components=n_components,
            weights=torch.softmax(torch.randn(n_components) * weight_scale, dim=0),
            seed=0,
            scale=1.0
        )
        super().__init__(
            x0=torch.randn(size=(n_chains, *target.event_shape)) * 2,
            target=target,
            **kwargs
        )


class DoubleShellBenchmark(Benchmark):
    def __init__(self, **kwargs):
        torch.random.fork_rng()
        torch.manual_seed(0)
        n_chains = 100
        target = DoubleGammaShell()
        super().__init__(
            x0=torch.randn(size=(n_chains, *target.event_shape)) * 2,
            target=target,
            **kwargs
        )


class SmallDoubleWellBenchmark(Benchmark):
    def __init__(self, **kwargs):
        torch.random.fork_rng()
        torch.manual_seed(0)
        n_chains = 100
        target = DoubleWell(event_shape=(10,))
        super().__init__(
            x0=torch.randn(size=(n_chains, *target.event_shape)) * 2,
            target=target,
            **kwargs
        )


class BigDoubleWellBenchmark(Benchmark):
    def __init__(self, event_shape: Tuple[int] = (100,), **kwargs):
        torch.random.fork_rng()
        torch.manual_seed(0)
        n_chains = 100
        target = DoubleWell(event_shape=event_shape)
        super().__init__(
            x0=torch.randn(size=(n_chains, *target.event_shape)) * 2,
            target=target,
            **kwargs
        )


if __name__ == '__main__':
    strategies = [
        ('jump_hmc', 'realnvp'),
        ('neutra_hmc', 'realnvp'),
        ('imh', 'realnvp'),
        'hmc',
    ]

    SeparatedMultimodalBenchmark(
        strategies=strategies,
        output_dir=pathlib.Path("output/separated_multimodal")
    ).run()

    OverlappingMultimodalBenchmark(
        strategies=strategies,
        output_dir=pathlib.Path("output/overlapping_multimodal")
    ).run()
