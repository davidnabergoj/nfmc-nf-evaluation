import os

LONG_WARMUP_TIME_MINUTES = 3 * 60
LONG_WARMUP_TIME_SECONDS = LONG_WARMUP_TIME_MINUTES * 60

MEDIUM_WARMUP_TIME_MINUTES = 30
MEDIUM_WARMUP_TIME_SECONDS = MEDIUM_WARMUP_TIME_MINUTES * 60

SHORT_WARMUP_TIME_MINUTES = 2
SHORT_WARMUP_TIME_SECONDS = SHORT_WARMUP_TIME_MINUTES * 60

INFINITE_WARMUP_ITERATIONS = 100_000_000_000

LONG_SAMPLING_TIME_MINUTES = 10 * 60
LONG_SAMPLING_TIME_SECONDS = LONG_SAMPLING_TIME_MINUTES * 60

MEDIUM_SAMPLING_TIME_MINUTES = 60
MEDIUM_SAMPLING_TIME_SECONDS = MEDIUM_SAMPLING_TIME_MINUTES * 60

SHORT_SAMPLING_TIME_MINUTES = 10
SHORT_SAMPLING_TIME_SECONDS = SHORT_SAMPLING_TIME_MINUTES * 60

INFINITE_SAMPLING_ITERATIONS = 100_000_000_000

DEFAULT_OVERHEAD_TIME_MINUTES = 10
DEFAULT_JOB_TIME_MINUTES = 60
DEFAULT_MEMORY_GB = 16
ROOT_DIR = f'{os.environ["USER_DIR"]}/nfmc-experiments/nfmc-nf-comparison'

# ARNES partitions
DEFAULT_PARTITION = None
DEFAULT_GPU_PARTITION = 'gpu'

DEFAULT_PYTHON_EXECUTABLE = (
    f'singularity '
    f'exec '
    f'{os.environ["USER_DIR"]}/environment.sif '
    f'/miniconda3/envs/torchenv/bin/python'
)

DEFAULT_PYTHONPATH = (
    f'export PYTHONPATH="{os.environ["USER_DIR"]}/nfmc-experiments/torchflows:'
    f'{os.environ["USER_DIR"]}/nfmc-experiments/nfmc:'
    f'{os.environ["USER_DIR"]}/nfmc-experiments/potentials:'
    f'{os.environ["USER_DIR"]}/nfmc-experiments/nfmc-nf-comparison:'
    f'{os.environ["USER_DIR"]}/nfmc-experiments/mcmc-diagnostics"'
)

SUPPORTED_NORMALIZING_FLOWS = [
    'c-lrsnsf',
    'c-naf-deep',
    'c-naf-deep-dense',
    'c-naf-dense',
    'c-rqnsf',
    'ddb',
    'ffjord',
    'i-resnet',
    'ia-lrsnsf',
    'ia-naf-deep',
    'ia-naf-deep-dense',
    'ia-naf-dense',
    'ia-rqnsf',
    'iaf',
    'ma-lrsnsf',
    'ma-naf-deep',
    'ma-naf-deep-dense',
    'ma-naf-dense',
    'ma-rqnsf',
    'maf',
    'nice',
    'planar',
    'radial',
    'realnvp',
    'resflow',
    'rnode',
    'sylvester'
]

GRADIENT_FREE_SAMPLERS = [
    'mh',
    'fixed_imh',
    'adaptive_imh',
    'neutra_mh',
    'jump_mh',
]

GRADIENT_BASED_SAMPLERS = [
    'hmc',
    'neutra_hmc',
    'jump_hmc',
]

MCMC_SAMPLERS = [
    'mh',
    'hmc',
]

NFMC_SAMPLERS = [
    'fixed_imh',
    'adaptive_imh',
    'neutra_mh',
    'jump_mh',
    'neutra_hmc',
    'jump_hmc',
]

NFMC_SAMPLERS_NON_ADAPTIVE = [
    'fixed_imh',
    'neutra_mh',
    'jump_mh',
    'neutra_hmc',
    'jump_hmc',
]

CONVOLUTIONAL_NORMALIZING_FLOWS = [
    'conv-rnode',
    'conv-ddb',
    'conv-ffjord',
    'conv-resflow',
    'conv-i-resnet',
    'glow-naf-dense',
    'ms-naf-dense',
    'glow-naf-deep-dense',
    'ms-naf-deep-dense',
    'glow-naf-deep',
    'ms-naf-deep',
    'glow-lrsnsf',
    'ms-lrsnsf',
    'glow-rqnsf',
    'ms-rqnsf',
    'glow-nice',
    'ms-nice',
    'glow-realnvp',
    'ms-realnvp',
]

TARGETS_SYNTHETIC_GAUSSIAN = [
    'standard_gaussian',
    'diagonal_gaussian',
    'full_rank_gaussian',
    'ill_conditioned_full_rank_gaussian',
]

TARGETS_SYNTHETIC_CURVED = [
    'rosenbrock',
    'funnel',
]

TARGETS_SYNTHETIC_MULTIMODAL = [
    'separated_multimodal',
    'overlapping_multimodal',
    'small_double_well',
    'big_double_well',
]

TARGETS_REAL_HIERARCHICAL = [
    'sparse_german_credit',
    'german_credit',
    'radon_intercepts_slopes',
    'radon_intercepts',
    'radon_slopes',
    'eight_schools',
    'synthetic_item_response_theory',
    'stochastic_volatility',
]

TARGETS_SYNTHETIC = TARGETS_SYNTHETIC_GAUSSIAN + TARGETS_SYNTHETIC_CURVED + TARGETS_SYNTHETIC_MULTIMODAL
TARGETS_NON_IMAGE = TARGETS_SYNTHETIC + TARGETS_REAL_HIERARCHICAL

TARGETS_IMAGE = [
    'phi4_8',
    'phi4_16',
    'phi4_32',
    'phi4_64',
    'phi4_128',
    'phi4_256',
]
