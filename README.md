# Normalizing flow evaluation in MCMC

This repository provides a benchmark and PyTorch framework for evaluating normalizing flows on Markov Chain Monte Carlo tasks.
The benchmark tests can be run locally or on a cluster with the SLURM framework.

It includes the following target distributions:
* Gaussian targets:
  * 100D standard Gaussian
  * 100D diagonal Gaussian
  * 100D full-rank Gaussian
  * 100D ill-conditioned full-rank Gaussian

* Unimodal non-Gaussian targets:
  * 100D Rosenbrock
  * 100D funnel

* Multimodal targets:
  * Two 100D Gaussian mixtures
  * Double-well,
  * Double Gamma shell

* Real-world posteriors:
  * Eight schools
  * German credit
  * Sparse German credit
  * Radon (varying intercepts)
  * Radon (varying slopes)
  * Radon (varying intercepts and slopes)
  * Synthetic item response theory
  * Stochastic volatility model

Running the notebooks with the appropriate dependency versions will reproduce experiments from the paper "Empirical evaluation of normalizing flows in Markov Chain Monte Carlo" by Nabergoj and Štrumbelj (2024).

The package was designed to be used with the supporting [torchflows](https://github.com/davidnabergoj/torchflows), [NFMC](https://github.com/davidnabergoj/nfmc), and [potentials](https://github.com/davidnabergoj/potentials) packages.
To test custom flow architectures, flow-based MCMC algorithms, or add new target distributions, please implement them using the framework provided in these packages.

## Citation

If you use this code in your work, we kindly ask that you cite the accompanying paper:
> [Nabergoj and Štrumbelj: Empirical evaluation of normalizing flows in Markov Chain Monte Carlo, 2024. arxiv:2412.17136.](https://arxiv.org/abs/2412.17136)

BibTex entry:
```
@misc{nabergoj_nf_mcmc_evaluation_2024,
    author = {Nabergoj, David and \v{S}trumbelj, Erik},
	title = {Empirical evaluation of normalizing flows in {Markov} {Chain} {Monte} {Carlo}},
	publisher = {arXiv},
	month = dec,
	year = {2024},
	note = {arxiv:2412.17136}
}
```


## Setup
We provide instructions on how to configure the repository for running the benchmark tests locally or on a cluster.

### Local setup
This package was tested with Python version 3.10.
We expect Python versions 3.7+ to also work, potentially with minor adjustments to dependency versions.

Clone the following supporting packages:
```
git clone git@github.com:davidnabergoj/torchflows.git
git clone git@github.com:davidnabergoj/potentials.git
git clone git@github.com:davidnabergoj/nfmc.git
git clone git@github.com:davidnabergoj/nfmc-nf-comparison.git
```

Install the required Python packages:
```
pip install -r requirements.txt
```

### SLURM setup
If running the benchmark tests on SLURM, follow the local setup by cloning the packages on the cluster.
Depending on the cluster, you may wish to create a Singularity environment with required packages.
We provide the `environment.def` file to build the Singularity environment.

Modify the following variables:
* `ROOT_DIR` in slurm/constants.py
* `DEFAULT_PYTHON_EXECUTABLE` in slurm/constants.py
* `DEFAULT_PYTHONPATH` in slurm/constants.py
* `DEFAULT_PARTITION` in slurm/constants.py

### Reproducibility
To reproduce paper results, use the following package commits:
* torchflows: ed213bd
* potentials: 2ecfe2f
* nfmc: 75ef6ce

## Running benchmark tests

To run benchmark tests locally, import the benchmark class from `benchmarking/gaussian.py`, `benchmarking/multimodal.py`, or `benchmarking/non_gaussian.py`:

```python
from benchmarking.gaussian import StandardGaussianBenchmark
import pathlib

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
```

To run tests on the cluster, run the `submit.py` scripts in subdirectories of `slurm/experiments`.

### Compiling and analyzing cluster outputs
We suggest compiling cluster results with the script `slurm/experiments/merge_data.py` and analyzing the resulting JSON file.
