import torch
from slurm.parsers import NFMCJobArgumentParser
from slurm.util import make_benchmark

if __name__ == '__main__':
    parser = NFMCJobArgumentParser()
    args = parser.parse_args()
    if args.device == 'gpu':
        args.device = 'cuda'
    benchmark = make_benchmark(args)
    torch.manual_seed(0)
    benchmark.run(device=args.device)
