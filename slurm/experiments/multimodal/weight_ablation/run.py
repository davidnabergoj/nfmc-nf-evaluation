import pathlib

import torch
from slurm.parsers import NFMCJobArgumentParser
from slurm.util import get_benchmark_class

if __name__ == '__main__':
    parser = NFMCJobArgumentParser()
    parser.add_argument('--weight_scale', type=float)
    args = parser.parse_args()
    if args.device == 'gpu':
        args.device = 'cuda'

    benchmark_class = get_benchmark_class(args.benchmark)
    benchmark = benchmark_class(
        weight_scale=args.weight_scale,
        strategies=[(args.sampler, args.flow)],
        output_dir=pathlib.Path(args.results_file).parent,
        sampling_time_limit_seconds=args.sampling_time_limit_seconds,
        n_sampling_iterations=args.sampling_iterations,
        warmup_time_limit_seconds=args.warmup_time_limit_seconds,
        n_warmup_iterations=args.warmup_iterations,
        output_files=[args.results_file],
        extra_data=dict(weight_scale=args.weight_scale),
    )

    torch.manual_seed(0)
    benchmark.run(device=args.device)
