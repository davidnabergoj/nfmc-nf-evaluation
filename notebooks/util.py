import ast
from pathlib import Path
from typing import Iterable, List, Union, Dict

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

from nfmc.util import get_flow_family, get_supported_normalizing_flows


def replace(string, replacements: Dict[str, str]):
    for old, new in replacements.items():
        string = string.replace(old, new)
    return string


FLOW_PRETTY = {
    "c-lrsnsf": 'C-LR-NSF',
    "c-naf-deep": 'C-NAF (deep)',
    "c-naf-deep-dense": 'C-NAF (deep-dense)',
    "c-naf-dense": 'C-NAF (dense)',
    "c-rqnsf": 'C-RQ-NSF',
    "ddb": 'CNF (Euler)',
    "ffjord": 'CNF (RK;D/T)',
    "rnode": 'CNF (RK;S/W)',
    "i-resnet": 'i-ResNet',
    "nice": 'NICE',
    "realnvp": 'RealNVP',
    "resflow": 'ResFlow',
    "p-resflow": 'Proximal ResFlow',
    "ia-lrsnsf": 'IA-LR-NSF',
    "ia-naf-deep": 'IA-NAF (deep)',
    "ia-naf-deep-dense": 'IA-NAF (deep-dense)',
    "ia-naf-dense": 'IA-NAF (dense)',
    "iaf": 'IAF',
    "planar": 'Planar',
    "sylvester": 'Sylvester',
    "radial": 'Radial',
    "ia-rqnsf": 'IA-RQ-NSF',
    "ia-rqsnsf": 'IA-RQ-NSF',
    "ot-flow": 'OT-flow',
}

FLOW_PRETTY_MATH = {
    k: replace(v, {
        'RealNVP': 'Real NVP',
        'NAF (deep-dense)': 'NAF$_\\mathrm{both}$',
        'NAF (dense)': 'NAF$_\\mathrm{dense}$',
        'NAF (deep)': 'NAF$_\\mathrm{deep}$',
        'CNF (Euler)': 'CNF$_\\mathrm{Euler}$',
        'CNF (RK;D/T)': 'CNF$_\\mathrm{RK}$',
        'CNF (RK;S/W)': 'CNF$_\\mathrm{RK(R)}$',
    })
    for k, v in FLOW_PRETTY.items()
}

FLOW_FAMILY_PRETTY = {
    f: f.capitalize()
    for f in ['autoregressive', 'residual', 'continuous']
}

SAMPLER_PRETTY = {
    'mh': 'MH',
    'imh': 'IMH',
    'fixed_imh': 'Fixed IMH',
    'adaptive_imh': 'Adaptive IMH',
    'neutra_mh': 'NeuTra MH',
    'jump_mh': 'Jump MH',
    'hmc': 'HMC',
    'jump_hmc': 'Jump HMC',
    'neutra_hmc': 'NeuTra HMC',
}


def get_standard_sampler_order(samplers: Union[List[str], np.array]):
    # Do not pass samplers as a pd.Series! Use samplers.values in that case.
    tmp = pd.DataFrame({'sampler': samplers})
    tmp['order'] = np.arange(len(samplers))
    tmp['sampler'] = tmp['sampler'].map({
        'mh': 0,
        'imh': 1,
        'fixed_imh': 1,
        'adaptive_imh': 1.2,
        'jump_mh': 1.5,
        'neutra_mh': 2,

        'hmc': 4,
        'jump_hmc': 5,
        'neutra_hmc': 6,
    })
    tmp.sort_values('sampler', inplace=True, ascending=True)
    return tmp['order'].values


def get_standard_flow_order(flows: Union[List[str], np.array]):
    # Do not pass flows as a pd.Series! Use flows.values in that case.
    tmp = pd.DataFrame({'flow': flows})
    tmp['order'] = np.arange(len(flows))
    tmp = pd.concat([tmp, pd.DataFrame([get_flow_families(f) for f in flows])], axis=1)
    tmp['flow_family'] = tmp['flow_family'].map({
        'autoregressive': 0,
        'residual': 1,
        'continuous': 2,
    })
    tmp['transformer_family'] = tmp['transformer_family'].map({
        'affine': 0,
        'spline': 1,
        'nn': 2
    })
    tmp['is_both_naf'] = tmp['flow'].map(lambda s: 'naf-deep-dense' in s)
    tmp['is_deep_naf'] = tmp['flow'].map(lambda s: s.endswith('naf-deep'))
    tmp['is_dense_naf'] = tmp['flow'].map(lambda s: 'naf-dense' in s)
    tmp.sort_values([
        'flow_family',
        'flow_subfamily',
        'convolutional',
        'transformer_family',
        'is_both_naf',
        'is_dense_naf',
        'is_deep_naf',
        'flow',
    ],
        inplace=True,
        ascending=True
    )
    return tmp['order'].values

BENCHMARK_FAMILY_ORDER = {
    'gaussian': 0,
    'non-gaussian (curved)': 1,
    'multimodal': 2,
    'non-gaussian (hierarchical)': 3,
}

BENCHMARK_FAMILY_ORDERED_LIST = ['gaussian', 'non-gaussian (curved)', 'multimodal', 'non-gaussian (hierarchical)']

BENCHMARK_FAMILY_PRETTY = {
    'gaussian': 'Gaussian',
    'multimodal': 'Multimodal',
    'non-gaussian (curved)': 'Non-Gaussian',
    'non-gaussian (hierarchical)': 'Real-world',
}

BENCHMARK_PRETTY = {
    'standard_gaussian': 'Standard Gaussian',
    'diagonal_gaussian': 'Diagonal Gaussian',
    'full_rank_gaussian': 'Full Rank Gaussian',
    'ill_conditioned_full_rank_gaussian': 'Ill-conditioned Full Rank Gaussian',

    'small_double_well': 'Double Well (10D)',
    'big_double_well': 'Double Well (100D)',
    'overlapping_multimodal': 'Overlapping multimodal',
    'separated_multimodal': 'Separated multimodal',

    'rosenbrock': 'Rosenbrock',
    'funnel': 'Funnel',

    'eight_schools': 'Eight schools',
    'german_credit': 'German credit',
    'sparse_german_credit': 'Sparse German credit',
    'phi4_small': '$\\phi^4$ (L = 8)',
    'phi4_big': '$\\phi^4$ (L = 64)',

    'radon_intercepts': 'Radon (intercepts)',
    'radon_slopes': 'Radon (slopes)',
    'radon_intercepts_slopes': 'Radon (intercepts and slopes)',

    'synthetic_item_response_theory': 'Synthetic Item Response Theory',
    'stochastic_volatility': 'Stochastic Volatility',
}


def get_benchmark_family(benchmark: str):
    benchmark_family = 'non-gaussian (hierarchical)'
    if 'gaussian' in benchmark:
        benchmark_family = 'gaussian'
    elif benchmark in ['big_double_well', 'small_double_well', 'double_shell',
                       'overlapping_multimodal', 'separated_multimodal']:
        benchmark_family = 'multimodal'
    elif benchmark in ['rosenbrock', 'funnel']:
        benchmark_family = 'non-gaussian (curved)'
    return benchmark_family


def get_flow_families(flow: str):
    flow_family = None
    flow_subfamily = None
    transformer_family = None
    convolutional = False
    if flow is not None:
        flow_family_data = get_flow_family(flow)
        flow_family = flow_family_data[0]
        flow_subfamily = flow_family_data[1]
        convolutional = False
        if flow_family == 'autoregressive':
            transformer_family = flow_family_data[2]
            convolutional = flow_subfamily == 'multiscale'
        if flow_family == 'residual':
            transformer_family = None
            convolutional = (flow_subfamily == 'iterative') and (flow_family_data[2] == 'convolutional')
        if flow_family == 'continuous':
            convolutional = flow_subfamily == 'convolutional'
    return {
        'flow_family': flow_family,
        'flow_subfamily': flow_subfamily,
        'convolutional': convolutional,
        'transformer_family': transformer_family,
    }


def standardized_rank(df,
                      rank_what: Iterable[str] = ('flow',),
                      metric: str = 'second_moment_squared_bias',
                      rank_across: Iterable[str] = ('benchmark',),
                      summary: str = 'median'):
    """
    Consider all benchmarks in df_subset.
    For each benchmark, compute the standardized rank of every architecture.
    Return the empirical mean of standardized ranks for every architecture across all benchmarks and the estimated standard error of the mean.
    If only one benchmark is present in df_subset, the standard error of the mean becomes nan.
    """
    if type(rank_what) == str:
        rank_what = (rank_what,)
    if type(rank_across) == str:
        rank_across = (rank_across,)

    if isinstance(rank_what, tuple):
        rank_what = list(rank_what)
    if isinstance(rank_across, tuple):
        rank_across = list(rank_across)

    ranks = []
    for _, rank_group in df[rank_across].drop_duplicates().iterrows():
        mask = np.ones(shape=(df.shape[0],), dtype=bool)
        for i in range(len(rank_across)):
            mask &= np.asarray(df[rank_across[i]].values == rank_group.iloc[i])
        tmp = df[mask][[*rank_what, metric]].groupby(rank_what)

        if summary == 'median':
            tmp = tmp.median()
        elif summary == 'mean':
            tmp = tmp.mean()
        elif summary == 'min':
            tmp = tmp.min()
        elif summary == 'max':
            tmp = tmp.max()
        elif summary == 'var':
            tmp = tmp.var()
        else:
            raise ValueError

        tmp = tmp.sort_values(metric).reset_index()
        data_point = dict()
        data_point_values = []
        rank = 0
        for index, row in tmp.iterrows():
            try:
                tuple_key = tuple([v for v in row[rank_what].values])
                data_point[str(tuple_key)] = rank
                data_point_values.append(rank)
                rank += 1
            except TypeError:
                continue

        if len(data_point_values) == 0:
            continue
        data_point_values = np.array(data_point_values)
        mu = float(data_point_values.mean())
        std = float(data_point_values.std())
        if len(data_point_values) == 1:
            std = 1.0
        standardized_data_point = {k: v if type(v) == str else (v - mu) / std for k, v in data_point.items()}
        ranks.append(standardized_data_point)
    tmp = pd.DataFrame(ranks).T
    tmp.index = pd.MultiIndex.from_tuples(tmp.index.map(ast.literal_eval), names=rank_what)

    average_rank = tmp.mean(axis='columns', numeric_only=True)
    sem_rank = tmp.sem(axis='columns', numeric_only=True)
    out = pd.DataFrame({
        'mean_rank': average_rank,
        'sem_rank': sem_rank
    }, index=tmp.index)
    return out.sort_index()


def make_data_entry(metric_name: str, metric_value: float, stem: str):
    alg = stem.split('-')[0]
    target = stem.split('-')[1]
    n_dim = int(stem.split('-')[-1])
    flow = '-'.join(stem.split('-')[2:-1])
    return dict(
        flow=flow,
        alg=alg,
        target=target,
        n_dim=n_dim,
        metric_value=metric_value,
        metric_name=metric_name
    )


def parse_subdirectory(subdirectory: Path, metric_name: str):
    for file_path in tqdm(subdirectory.glob('*.*'), desc=f'Parsing {metric_name}'):
        if "rhat" in metric_name:
            values = torch.load(file_path)
            new_value = float(torch.mean(values))
            new_name = f"mean_{metric_name}"
        elif "bias2" == metric_name:
            values = torch.load(file_path)
            new_value = float(torch.min(values))
            new_name = f'min_{metric_name}'
        elif "process_time" == metric_name:
            elapsed_seconds = -1.0
            with open(file_path, "r") as f:
                for line in f.read().splitlines():
                    if 'elapsed' in line:
                        elapsed_seconds = float(line.split(':')[1])
            new_value = elapsed_seconds
            new_name = metric_name
        else:
            raise ValueError

        yield make_data_entry(
            metric_name=new_name,
            metric_value=new_value,
            stem=file_path.stem
        )


def to_booktabs_table(df,
                      precision: int = 2,
                      caption='Caption',
                      label='tab:my-table',
                      alignment='S',
                      bold_mask: np.ndarray = None,
                      save_to_file: str = None,
                      uncertainty_formatting: bool = True):
    lines = ['\\begin{table}']  # Lines to print
    if uncertainty_formatting:
        lines.append(
            r'''
            \renewrobustcmd{\bfseries}{\fontseries{b}\selectfont}
            \renewrobustcmd{\boldmath}{}
            \sisetup{%
                table-align-uncertainty=true,
                detect-all,
                separate-uncertainty=true,
                mode=text,
                round-mode=uncertainty,
                round-precision=2,
                table-format = 2.2(2),
                table-column-width=2.1cm
            }
            '''
        )
    column_specification = "\n".join(["l"] + [alignment] * (len(df.columns) - 1))
    lines.append(f'\\begin{{tabular}}{{{column_specification}}}')

    table_header_string = ' & '.join(map(lambda s: f'{{{s}}}', df.columns)) + ' \\\\'

    def format_cell(c):
        if isinstance(c, tuple):
            if np.isfinite(c[0]):
                if c[1] == 0.0:
                    return str(round(c[0], precision))
                elif np.isfinite(c[1]):
                    return f'{c[0]:.2f}({c[1]:.2f})'
                else:
                    return f'{round(c[0], precision):.2f}'
            else:
                return ''
        elif isinstance(c, float):
            if np.isnan(c):
                return ''
            return str(round(c, precision))
        elif isinstance(c, int):
            return str(c)
        else:
            return f'{{{c}}}'

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        new_row_data = []
        for j, col in enumerate(row):
            prefix = '' if bold_mask is None or not bold_mask[i, j] else '\\bfseries '
            new_row_data.append(prefix + format_cell(col))
        rows.append(new_row_data)
    table_rows_strings = [' & '.join(row) + ' \\\\' for row in rows]

    lines.append(f'\\toprule')
    lines.append(table_header_string)
    lines.append(f'\\midrule')
    lines.append('\n'.join(table_rows_strings))
    lines.append(f'\\bottomrule')

    lines.append(f'\\end{{tabular}}')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(f'\\end{{table}}')

    for line in lines:
        print(line)
    if save_to_file is not None:
        with open(save_to_file, 'w') as f:
            f.writelines(line + '\n' for line in lines)


def standardized_rank_best_nf_kwargs(df, **kwargs):
    # Get df with keys: values = flow: argmin_{r} flow_kwargs
    tmp = standardized_rank(
        df.astype({'flow_kwargs': str}),
        rank_what=['flow', 'flow_kwargs'],
        rank_across='benchmark'
    ).sort_values([
        'mean_rank'
    ]).reset_index().drop_duplicates(subset=['flow'])
    best_kwargs_df = tmp[['flow', 'flow_kwargs']].reset_index(drop=True)

    # Extract from df the rows which match the flow and argmin_{r} flow_kwargs pairs that we just computed
    tmp0 = df.reset_index(drop=True)
    tmp0['key'] = list(map(hash, tmp0['flow_kwargs'].astype(str).values + tmp0['flow'].values))
    best_kwargs_df['key'] = list(
        map(hash, best_kwargs_df['flow_kwargs'].values + best_kwargs_df['flow'].values)
    )
    tmp = best_kwargs_df[['key']].set_index('key').join(tmp0.set_index('key'), how='inner').reset_index(drop=True)
    return standardized_rank(tmp, **kwargs)


def make_bold_mask(_df, top_quantile: float = 0.9, prepend=True):
    _tmp = _df.apply(lambda r: [v[0] if isinstance(v, tuple) else v for v in r])
    limits = _tmp.quantile(1 - top_quantile)  # negate because we want the smallest values
    mask = _tmp <= limits
    if prepend:
        mask = np.c_[np.zeros(len(mask), dtype=bool), mask]
    return mask


def standardized_rank_multiple(*dfs, **kwargs):
    return pd.concat([standardized_rank(df, **kwargs) for df in dfs], axis=1)


def ecdf(inputs, dataset: np.array = None):
    if dataset is not None:
        return 1 / len(dataset) * np.sum(dataset <= inputs)
    else:
        return np.array([ecdf(i, inputs) for i in inputs])


def standardized_rank_multiple_masks(df,
                                     masks,
                                     new_names: List[str] = None,
                                     decimal_precision: int = 2,
                                     **kwargs):
    if new_names is None:
        new_names = [f'col{i}' for i in range(len(masks))]
    concatenated = pd.concat([
        standardized_rank(df[m], **kwargs).apply(lambda el: round(el, decimal_precision)).apply(tuple, axis=1)
        for m in masks
    ], axis=1)
    concatenated = concatenated.reset_index()
    concatenated.columns = [concatenated.columns[0]] + list(new_names)
    return concatenated


def standardized_rank_multiple_masks_by_col(df,
                                            col,
                                            include_all: bool = True,
                                            **kwargs):
    col_names = list(df[col].unique())
    masks = [df[col] == c for c in col_names]
    if include_all:
        masks.append(pd.Series(np.ones(df.shape[0], dtype=bool)).values)
        col_names.append('All')
    return standardized_rank_multiple_masks(df, masks, new_names=col_names, **kwargs)


def make_best_flow_kwargs_col(df):
    df['flow_kwargs'] = df['flow_kwargs'].map(str)
    # df = df[~df['sampler'].isin(['hmc', 'mh'])]

    from collections import defaultdict

    # Create column indicating the best

    best_kwargs_per_flow = defaultdict(str)
    for flow in df['flow'].unique():
        flow_subset = df[df['flow'] == flow]
        kwarg_win_counts = defaultdict(int)
        for index, row in flow_subset[['benchmark', 'sampler']].drop_duplicates().iterrows():
            benchmark = row['benchmark']
            sampler = row['sampler']
            experiment_subset = flow_subset[(
                    (flow_subset['sampler'] == sampler)
                    & (flow_subset['benchmark'] == benchmark)
            )]

            kwargs_string = experiment_subset['flow_kwargs'].values
            b2 = experiment_subset['second_moment_squared_bias'].values
            order = np.argsort(b2)
            kwarg_win_counts[kwargs_string[order][0]] += 1
        best_kwargs_per_flow[flow] = max(kwarg_win_counts, key=kwarg_win_counts.get)

    mask = []
    for index, row in df[['flow', 'flow_kwargs']].iterrows():
        if best_kwargs_per_flow[row['flow']] == row['flow_kwargs']:
            mask.append(True)
        else:
            mask.append(False)
    return mask  # were the best kwargs for the NF used in this experiment?
    # Best kwargs are determined by most top rankings across all experiments (for a NF)


if __name__ == '__main__':
    get_standard_flow_order(get_supported_normalizing_flows(synonyms=False))

    data_in = ['neutra_mh', 'jump_mh', 'imh', 'neutra_hmc', 'jump_hmc', 'hmc']
    print(np.array(data_in)[get_standard_sampler_order(data_in)])
