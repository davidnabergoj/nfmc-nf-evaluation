from typing import Optional

import time

import itertools
import json
import pathlib

from slurm.constants import GRADIENT_FREE_SAMPLERS, GRADIENT_BASED_SAMPLERS, MCMC_SAMPLERS, \
    CONVOLUTIONAL_NORMALIZING_FLOWS


def get_benchmark_class(benchmark_str):
    from benchmarking.gaussian import (
        StandardGaussianBenchmark,
        DiagonalGaussianBenchmark,
        FullRankGaussianBenchmark,
        IllConditionedFullRankGaussianBenchmark
    )
    from benchmarking.multimodal import (
        SeparatedMultimodalBenchmark,
        OverlappingMultimodalBenchmark,
        DoubleShellBenchmark,
        SmallDoubleWellBenchmark,
        BigDoubleWellBenchmark
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
        StochasticVolatilityModelBenchmark
    )
    from benchmarking.non_gaussian import (
        Phi4Side8,
        Phi4Side16,
        Phi4Side32,
        Phi4Side64,
        Phi4Side128,
        Phi4Side256,
    )
    return {
        'standard_gaussian': StandardGaussianBenchmark,
        'diagonal_gaussian': DiagonalGaussianBenchmark,
        'full_rank_gaussian': FullRankGaussianBenchmark,
        'ill_conditioned_full_rank_gaussian': IllConditionedFullRankGaussianBenchmark,
        'separated_multimodal': SeparatedMultimodalBenchmark,
        'overlapping_multimodal': OverlappingMultimodalBenchmark,
        'double_shell': DoubleShellBenchmark,
        'small_double_well': SmallDoubleWellBenchmark,
        'big_double_well': BigDoubleWellBenchmark,
        'rosenbrock': RosenbrockBenchmark,
        'biochemical_oxygen_demand': BiochemicalOxygenDemandBenchmark,
        'sparse_german_credit': SparseGermanCreditBenchmark,
        'german_credit': GermanCreditBenchmark,
        'basketball_v1': BasketballV1Benchmark,
        'basketball_v2': BasketballV2Benchmark,
        'funnel': FunnelBenchmark,
        'radon_intercepts_slopes': RadonVaryingInterceptsAndSlopesBenchmark,
        'radon_intercepts': RadonVaryingInterceptsBenchmark,
        'radon_slopes': RadonVaryingSlopesBenchmark,
        'eight_schools': EightSchoolsBenchmark,
        'synthetic_item_response_theory': SyntheticItemResponseTheoryBenchmark,
        'stochastic_volatility': StochasticVolatilityModelBenchmark,
        'phi4_8': Phi4Side8,
        'phi4_16': Phi4Side16,
        'phi4_32': Phi4Side32,
        'phi4_64': Phi4Side64,
        'phi4_128': Phi4Side128,
        'phi4_256': Phi4Side256,
    }[benchmark_str]


def get_phi4_standard_strategies_jhmc():
    return [
        NFMCStrategy(flow=f, sampler=s)
        for s, f in [
            ('jump_hmc', 'nice'),
            ('jump_hmc', 'realnvp'),
            ('jump_hmc', 'c-rqnsf'),
            ('jump_hmc', 'c-lrsnsf'),
            ('jump_hmc', 'c-naf-deep'),
            ('jump_hmc', 'c-naf-dense'),
            ('jump_hmc', 'c-naf-deep-dense'),
            ('jump_hmc', 'i-resnet'),
            ('jump_hmc', 'resflow'),
            ('jump_hmc', 'p-resflow'),
            ('jump_hmc', 'ffjord'),
            ('jump_hmc', 'rnode'),
            ('jump_hmc', 'ot-flow'),
            ('jump_hmc', 'ddb'),
        ]
    ]


def get_strategies_imh():
    return [
        NFMCStrategy(flow=f, sampler=s)
        for s, f in [
            # Want efficient forward and inverse
            ('imh', 'realnvp'),
            ('imh', 'nice'),
            ('imh', 'c-rqnsf'),
            ('imh', 'c-lrsnsf'),
            ('imh', 'c-naf-deep'),
            ('imh', 'c-naf-dense'),
            ('imh', 'c-naf-deep-dense'),
            ('imh', 'rnode'),
            ('imh', 'ot-flow'),
            ('imh', 'ffjord'),
            ('imh', 'ddb'),
            ('imh', 'resflow'),
            ('imh', 'p-resflow'),
            ('imh', 'i-resnet'),
        ]
    ]


def get_coupling_configs_non_image_broad():
    """
    List of coupling NF configurations with broad hyperparameters.
    """
    configs = []
    for flow_name in ['realnvp', 'nice', 'c-rqnsf', 'c-lrsnsf', 'c-naf-deep', 'c-naf-dense', 'c-naf-deep-dense']:
        for n_layers, (c_n_hidden, c_n_layers) in itertools.product([2, 5, 10], [(10, 2), (100, 5)]):
            kwarg_dict = {'n_layers': n_layers, 'conditioner_kwargs': {'n_hidden': c_n_hidden, 'n_layers': c_n_layers}}
            configs.append(f'{flow_name}%{json.dumps(kwarg_dict).replace(" ", "")}')
    return configs


def get_inverse_autoregressive_configs_non_image_broad():
    """
    List of coupling NF configurations with broad hyperparameters.
    """
    configs = []
    for flow_name in [
        'iaf',
        'ia-rqnsf',
        'ia-lrsnsf',
        'ia-naf-dense',
        'ia-naf-deep',
        'ia-naf-deep-dense',
    ]:
        for n_layers, (c_n_hidden, c_n_layers) in itertools.product([2, 5, 10], [(10, 2), (100, 5)]):
            kwarg_dict = {'n_layers': n_layers, 'conditioner_kwargs': {'n_hidden': c_n_hidden, 'n_layers': c_n_layers}}
            configs.append(f'{flow_name}%{json.dumps(kwarg_dict).replace(" ", "")}')
    return configs


def get_residual_matdet_configs_non_image():
    configs = []
    for flow_name in [
        'planar',
        'radial',
        'sylvester',
    ]:
        for n_layers in [2, 5, 10]:
            kwarg_dict = {'n_layers': n_layers}
            configs.append(f'{flow_name}%{json.dumps(kwarg_dict).replace(" ", "")}')
    return configs


def get_continuous_configs_non_image_broad():
    configs = []
    for flow_name in ['rnode', 'ffjord', 'ddb']:
        for hidden_size, n_hidden_layers in itertools.product([10, 100], [1, 5, 10]):
            kwarg_dict = {"nn_kwargs": {"hidden_size": hidden_size, "n_hidden_layers": n_hidden_layers}}
            configs.append(f'{flow_name}%{json.dumps(kwarg_dict).replace(" ", "")}')
    for ot_hidden_size, resnet_hidden_size in itertools.product([10, 100], [10, 20, 100]):
        kwarg_dict = {
            "ot_flow_kwargs": {
                "hidden_size": ot_hidden_size,
                "resnet_kwargs": {"hidden_size": resnet_hidden_size}
            }
        }
        configs.append(f'ot-flow%{json.dumps(kwarg_dict).replace(" ", "")}')
    return configs


def get_residual_configs_non_image_broad():
    configs = []
    for flow_name in ['resflow', 'p-resflow', 'i-resnet']:
        for n_layers, (b_hidden_size, b_n_layers) in itertools.product([2, 5, 10], [(10, 2), (100, 5)]):
            kwarg_dict = {
                'n_layers': n_layers,
                'layer_kwargs': {'hidden_size': b_hidden_size, 'n_hidden_layers': b_n_layers}
            }
            configs.append(f'{flow_name}%{json.dumps(kwarg_dict).replace(" ", "")}')
    for flow_name in ['p-resflow']:
        for n_layers, (b_hidden_size, b_n_layers) in itertools.product([2, 5, 10], [(10, 2), (100, 5)]):
            kwarg_dict = {
                'n_layers': n_layers,
                'layer_kwargs': {'hidden_size': b_hidden_size, 'n_layers': b_n_layers}
            }
            configs.append(f'{flow_name}%{json.dumps(kwarg_dict).replace(" ", "")}')
    return configs


def get_strategies_imh_broad():
    # Want efficient forward and inverse
    # add hyperparameter variety
    # 6 times as many strategies
    strategies = []

    for config in (
            get_coupling_configs_non_image_broad()
            + get_residual_configs_non_image_broad()
            + get_continuous_configs_non_image_broad()
    ):
        strategies.append(NFMCStrategy(sampler='imh', flow=config))  # fixed imh
    return strategies


def get_strategies_adaptive_imh_broad():
    # Want efficient forward and inverse
    # add hyperparameter variety
    # 6 times as many strategies
    strategies = []

    for config in (
            get_coupling_configs_non_image_broad()
            + get_residual_configs_non_image_broad()
            + get_continuous_configs_non_image_broad()
    ):
        strategies.append(NFMCStrategy(sampler='adaptive_imh', flow=config))
    return strategies


def get_strategies_jhmc_broad():
    # Want efficient forward and inverse
    # add hyperparameter variety
    # 6 times as many strategies
    strategies = []

    for config in (
            get_coupling_configs_non_image_broad()
            + get_residual_configs_non_image_broad()
            + get_continuous_configs_non_image_broad()
    ):
        strategies.append(NFMCStrategy(sampler='jump_hmc', flow=config))
    return strategies


def get_strategies_jmh_broad():
    # Want efficient forward and inverse
    # add hyperparameter variety
    # 6 times as many strategies
    strategies = []

    for config in (
            get_coupling_configs_non_image_broad()
            + get_residual_configs_non_image_broad()
            + get_continuous_configs_non_image_broad()
    ):
        strategies.append(NFMCStrategy(sampler='jump_mh', flow=config))
    return strategies


def get_strategies_jhmc():
    return [
        NFMCStrategy(flow=f, sampler=s)
        for s, f in [
            # Want efficient forward and inverse
            ('jump_hmc', 'realnvp'),
            ('jump_hmc', 'nice'),
            ('jump_hmc', 'c-rqnsf'),
            ('jump_hmc', 'c-lrsnsf'),
            ('jump_hmc', 'c-naf-deep'),
            ('jump_hmc', 'c-naf-dense'),
            ('jump_hmc', 'c-naf-deep-dense'),
            ('jump_hmc', 'rnode'),
            ('jump_hmc', 'ot-flow'),
            ('jump_hmc', 'ffjord'),
            ('jump_hmc', 'ddb'),
            ('jump_hmc', 'resflow'),
            ('jump_hmc', 'p-resflow'),
            ('jump_hmc', 'i-resnet'),
        ]
    ]


def get_strategies_jmh():
    return [
        NFMCStrategy(flow=f, sampler=s)
        for s, f in [
            # Want efficient forward and inverse
            ('jump_mh', 'realnvp'),
            ('jump_mh', 'nice'),
            ('jump_mh', 'c-rqnsf'),
            ('jump_mh', 'c-lrsnsf'),
            ('jump_mh', 'c-naf-deep'),
            ('jump_mh', 'c-naf-dense'),
            ('jump_mh', 'c-naf-deep-dense'),
            ('jump_mh', 'rnode'),
            ('jump_mh', 'ot-flow'),
            ('jump_mh', 'ffjord'),
            ('jump_mh', 'ddb'),
            ('jump_mh', 'resflow'),
            ('jump_mh', 'p-resflow'),
            ('jump_mh', 'i-resnet'),
        ]
    ]


def get_strategies_neutra_hmc_broad():
    # Want efficient forward and inverse
    # add hyperparameter variety
    # 6 times as many strategies
    strategies = []

    for config in (
            get_coupling_configs_non_image_broad()
            + get_inverse_autoregressive_configs_non_image_broad()
            + get_residual_configs_non_image_broad()
            + get_residual_matdet_configs_non_image()
            + get_continuous_configs_non_image_broad()
    ):
        strategies.append(NFMCStrategy(sampler='neutra_hmc', flow=config))
    return strategies


def get_strategies_neutra_mh_broad():
    # Want efficient forward and inverse
    # add hyperparameter variety
    # 6 times as many strategies
    strategies = []

    for config in (
            get_coupling_configs_non_image_broad()
            + get_inverse_autoregressive_configs_non_image_broad()
            + get_residual_configs_non_image_broad()
            + get_residual_matdet_configs_non_image()
            + get_continuous_configs_non_image_broad()
    ):
        strategies.append(NFMCStrategy(sampler='neutra_mh', flow=config))
    return strategies


def get_strategies_neutra_mh():
    return [
        NFMCStrategy(flow=f, sampler=s)
        for s, f in [
            # Want efficient inverse
            ('neutra_mh', 'realnvp'),
            ('neutra_mh', 'nice'),
            ('neutra_mh', 'iaf'),
            ('neutra_mh', 'ia-rqnsf'),
            ('neutra_mh', 'ia-lrsnsf'),
            ('neutra_mh', 'ia-naf-dense'),
            ('neutra_mh', 'ia-naf-deep'),
            ('neutra_mh', 'ia-naf-deep-dense'),
            ('neutra_mh', 'c-rqnsf'),
            ('neutra_mh', 'c-lrsnsf'),
            ('neutra_mh', 'c-naf-deep'),
            ('neutra_mh', 'c-naf-dense'),
            ('neutra_mh', 'c-naf-deep-dense'),
            ('neutra_mh', 'rnode'),
            ('neutra_mh', 'ot-flow'),
            ('neutra_mh', 'ffjord'),
            ('neutra_mh', 'ddb'),
            ('neutra_mh', 'resflow'),
            ('neutra_mh', 'p-resflow'),
            ('neutra_mh', 'i-resnet'),
            ('neutra_mh', 'planar'),
            ('neutra_mh', 'radial'),
            ('neutra_mh', 'sylvester')
        ]
    ]


def get_strategies_neutra_hmc():
    return [
        NFMCStrategy(flow=f, sampler=s)
        for s, f in [
            # Want efficient inverse
            ('neutra_hmc', 'realnvp'),
            ('neutra_hmc', 'nice'),
            ('neutra_hmc', 'iaf'),
            ('neutra_hmc', 'ia-rqnsf'),
            ('neutra_hmc', 'ia-lrsnsf'),
            ('neutra_hmc', 'ia-naf-dense'),
            ('neutra_hmc', 'ia-naf-deep'),
            ('neutra_hmc', 'ia-naf-deep-dense'),
            ('neutra_hmc', 'c-rqnsf'),
            ('neutra_hmc', 'c-lrsnsf'),
            ('neutra_hmc', 'c-naf-deep'),
            ('neutra_hmc', 'c-naf-dense'),
            ('neutra_hmc', 'c-naf-deep-dense'),
            ('neutra_hmc', 'rnode'),
            ('neutra_hmc', 'ot-flow'),
            ('neutra_hmc', 'ffjord'),
            ('neutra_hmc', 'ddb'),
            ('neutra_hmc', 'resflow'),
            ('neutra_hmc', 'p-resflow'),
            ('neutra_hmc', 'i-resnet'),
            ('neutra_hmc', 'planar'),
            ('neutra_hmc', 'radial'),
            ('neutra_hmc', 'sylvester')
        ]
    ]


def get_strategies_broad_gradient_free_rest():
    return [
        NFMCStrategy(sampler='mh', flow=None),
        *get_strategies_neutra_mh_broad(),
        *get_strategies_jmh_broad(),
        *get_strategies_adaptive_imh_broad(),
    ]


def get_strategies_single_config():
    return [
        NFMCStrategy(sampler='hmc', flow=None),
        NFMCStrategy(sampler='mh', flow=None),
        *get_strategies_imh(),
        *get_strategies_jmh(),
        *get_strategies_neutra_mh(),
        *get_strategies_jhmc(),
        *get_strategies_neutra_hmc(),
    ]


def get_strategies_all():
    return [
        NFMCStrategy(sampler='hmc', flow=None),
        NFMCStrategy(sampler='mh', flow=None),
        *get_strategies_imh_broad(),
        *get_strategies_jmh_broad(),
        *get_strategies_neutra_mh_broad(),
        *get_strategies_jhmc_broad(),
        *get_strategies_neutra_hmc_broad(),
    ]


def get_strategies(include_gradient_free: bool = True,
                   include_gradient_based: bool = True,
                   include_fixed_imh: bool = True,
                   include_adaptive_imh: bool = True,
                   include_jump: bool = True,
                   include_neutra: bool = True,
                   include_mcmc: bool = True,
                   include_convolutional: bool = True):
    strategies = get_strategies_all()
    keep_strategy = [True] * len(strategies)

    if include_gradient_free:
        keep_strategy = [m & (s.sampler in GRADIENT_FREE_SAMPLERS) for m in keep_strategy for s in strategies]
    if include_gradient_based:
        keep_strategy = [m & (s.sampler in GRADIENT_BASED_SAMPLERS) for m in keep_strategy for s in strategies]
    if include_fixed_imh:
        keep_strategy = [m & (s.sampler == 'fixed_imh') for m in keep_strategy for s in strategies]
    if include_adaptive_imh:
        keep_strategy = [m & (s.sampler == 'adaptive_imh') for m in keep_strategy for s in strategies]
    if include_jump:
        keep_strategy = [m & ('jump' in s.sampler) for m in keep_strategy for s in strategies]
    if include_neutra:
        keep_strategy = [m & ('neutra' in s.sampler) for m in keep_strategy for s in strategies]
    if include_mcmc:
        keep_strategy = [m & (s.sampler in MCMC_SAMPLERS) for m in keep_strategy for s in strategies]
    if include_convolutional:
        keep_strategy = [
            m & (s.flow in CONVOLUTIONAL_NORMALIZING_FLOWS)
            if s.flow is not None else False
            for m in keep_strategy
            for s in strategies
        ]
    return [strategies[i] for i in range(len(strategies)) if keep_strategy[i]]


def parse_flow_string(flow_string: str):
    """
    Flow string syntax: <flow_name>%<json_string> or <flow_name>.
    """
    if flow_string is None:
        return {
            'name': None,
            'kwargs': {},
            'hash': hash('None')
        }

    if '%' not in flow_string:
        return {
            'name': flow_string,
            'kwargs': {},
            'hash': hash(flow_string)
        }
    else:
        flow_name = flow_string.split('%')[0]
        kwargs = json.loads(flow_string.split('%')[1])
        return {
            'name': flow_name,
            'kwargs': kwargs,
            'hash': hash(flow_name + str(kwargs))
        }


def make_benchmark(args):
    # args from NFMCJobArgumentParser
    benchmark_class = get_benchmark_class(args.benchmark)
    return benchmark_class(
        name=args.benchmark,
        strategies=[(args.sampler, args.flow)],
        output_dir=pathlib.Path(args.results_file).parent,
        sampling_time_limit_seconds=args.sampling_time_limit_seconds,
        n_sampling_iterations=args.sampling_iterations,
        warmup_time_limit_seconds=args.warmup_time_limit_seconds,
        n_warmup_iterations=args.warmup_iterations,
        output_files=[args.results_file],
    )


def print_submit_countdown(seconds: int = 3):
    assert seconds > 0
    print(f'Submitting in {seconds}, ', end='', flush=True)
    time.sleep(1)
    for i in range(seconds - 1, 0, -1):
        print(f'{i}, ', end='', flush=True)
        time.sleep(1)
    print('0', flush=True)


class NFMCStrategy:
    def __init__(self, sampler: str, flow: Optional[str]):
        self.sampler = sampler
        self.flow = flow
        if self.flow is None:
            self.flow = 'None'
