import pathlib

import pytest
import torch

from benchmarking.gaussian import (
    StandardGaussianBenchmark,
    IllConditionedFullRankGaussianBenchmark,
    DiagonalGaussianBenchmark,
    FullRankGaussianBenchmark
)

from benchmarking.non_gaussian import (
    RosenbrockBenchmark,
    BiochemicalOxygenDemandBenchmark,
    SparseGermanCreditBenchmark,
    GermanCreditBenchmark,
    BasketballV1Benchmark,
    BasketballV2Benchmark,
    FunnelBenchmark,
    RadonVaryingInterceptsAndSlopesBenchmark,
    RadonVaryingInterceptsBenchmark,
    RadonVaryingSlopesBenchmark,
    EightSchoolsBenchmark,
    SyntheticItemResponseTheoryBenchmark,
    StochasticVolatilityModelBenchmark,
)

from benchmarking.multimodal import (
    SeparatedMultimodalBenchmark,
    OverlappingMultimodalBenchmark,
    SmallDoubleWellBenchmark,
    BigDoubleWellBenchmark,
    DoubleShellBenchmark,
)


@pytest.mark.parametrize('benchmark', [
    StandardGaussianBenchmark,
    IllConditionedFullRankGaussianBenchmark,
    DiagonalGaussianBenchmark,
    FullRankGaussianBenchmark,
])
def test_gaussian(benchmark):
    torch.manual_seed(0)
    benchmark_obj = benchmark(
        strategies=[('jump_hmc', 'realnvp')],
        n_chains=4,
        n_iterations=10,
        warmup=False
    )
    benchmark_obj.run()


@pytest.mark.parametrize('benchmark', [
    RosenbrockBenchmark,
    BiochemicalOxygenDemandBenchmark,
    SparseGermanCreditBenchmark,
    GermanCreditBenchmark,
    # BasketballV1Benchmark,
    # BasketballV2Benchmark,
    FunnelBenchmark,
    RadonVaryingInterceptsAndSlopesBenchmark,
    RadonVaryingInterceptsBenchmark,
    RadonVaryingSlopesBenchmark,
    EightSchoolsBenchmark,
    SyntheticItemResponseTheoryBenchmark,
    StochasticVolatilityModelBenchmark,
])
def test_non_gaussian(benchmark):
    torch.manual_seed(0)
    benchmark_obj = benchmark(
        strategies=[('jump_hmc', 'realnvp')],
        n_chains=4,
        n_iterations=10,
        warmup=False
    )
    benchmark_obj.run()


@pytest.mark.parametrize('benchmark', [
    SeparatedMultimodalBenchmark,
    OverlappingMultimodalBenchmark,
    DoubleShellBenchmark,
    SmallDoubleWellBenchmark,
    BigDoubleWellBenchmark,
])
def test_multimodal(benchmark):
    torch.manual_seed(0)
    benchmark_obj = benchmark(
        strategies=[('jump_hmc', 'realnvp')],
        n_chains=4,
        n_iterations=10,
        warmup=False
    )
    benchmark_obj.run()


def test_save_results():
    import json

    torch.manual_seed(0)
    output_file = '____test_output.json'
    b = FunnelBenchmark(
        strategies=[('jump_hmc', 'realnvp')],
        output_files=[output_file],
        n_chains=4,
        n_iterations=10,
        warmup=False,
        extra_data={
            'c': 1.23
        }
    )
    b.run()
    with open(output_file, 'r') as f:
        contents = json.load(f)
    assert contents['c'] == 1.23
    
    pathlib.Path(output_file).unlink(missing_ok=True)
