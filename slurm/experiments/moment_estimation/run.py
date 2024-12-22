import pathlib
from argparse import ArgumentParser

import torch

from nfmc import sample
from slurm.util import get_benchmark_class

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--sampling_time_limit_seconds', type=float)
    parser.add_argument('--warmup_time_limit_seconds', type=float)
    parser.add_argument('--output_dir', type=str, default='moments')
    args = parser.parse_args()
    if args.device == 'gpu':
        args.device = 'cuda'

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_class = get_benchmark_class(args.benchmark)
    benchmark = benchmark_class(
        strategies=[],
        output_dir=output_dir / args.benchmark,
    )

    torch.manual_seed(0)
    output_file = output_dir / f'{args.benchmark}_moments.pt'
    output = sample(
        target=benchmark.target,
        strategy='hmc',
        n_iterations=100_000_000_000_000_000,
        n_warmup_iterations=100_000_000_000_000_000,
        warmup=True,
        param_kwargs={'store_samples': False},
        sampling_time_limit_seconds=args.sampling_time_limit_seconds,
        warmup_time_limit_seconds=args.warmup_time_limit_seconds,
    )
    torch.save(torch.stack([output.mean, output.second_moment]), output_file)
    # print(output.mean)
    # print(output.second_moment)
    print(f'Saving moments to: {output_file}')
