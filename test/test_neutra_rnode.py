def test_basic():
    from benchmarking.gaussian import StandardGaussianBenchmark

    StandardGaussianBenchmark(
        strategies=[
            ('neutra_mh', 'rnode'),
            ('neutra_hmc', 'rnode')
        ],
        n_warmup_iterations=3,
        n_sampling_iterations=3
    ).run()
