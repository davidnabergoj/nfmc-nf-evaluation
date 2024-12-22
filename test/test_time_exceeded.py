import torch
from benchmarking.non_gaussian import Phi4Side64


def test_sampling():
    torch.manual_seed(0)
    b = Phi4Side64(strategies=[('hmc', None)], sampling_time_limit_seconds=1.0, warmup=False)
    b.run()


def test_warmup():
    torch.manual_seed(0)
    b = Phi4Side64(strategies=[('hmc', None)], warmup_time_limit_seconds=1.0, n_sampling_iterations=1, warmup=True)
    b.run()


def test_both():
    torch.manual_seed(0)
    b = Phi4Side64(
        strategies=[('hmc', None)],
        sampling_time_limit_seconds=1.0,
        warmup_time_limit_seconds=1.0,
        warmup=True
    )
    b.run()
