#!/usr/bin/env python
# coding: utf-8
"""Configuration for DPQC and Unitary-PQC computations and plots."""


# ------------------------------------------------------------
# VQE optimization settings
# ------------------------------------------------------------
H_PARAM = 0.10
# R: number of independent VQE optimization trials for every layer.
NUM_RUNS = 100
STEPS = 3000
SAMPLE_EVERY = 1000
# Parameter update used by the DPQC and reset-DPQC VQE pipelines.
# Supported values are:
#   "adam": Adam (the historical default),
#   "gradient_descent": deterministic full-batch gradient descent.
DPQC_VQE_OPTIMIZER = "gradient_descent"
LEARNING_RATE = 1e-3
TOLERANCE = 1e-2
VQE_SEED_BASE = 0
# Independent VQE trials are compiled and evaluated in fixed-size batches.
# Five keeps the density-matrix/autodiff working set modest while removing the
# per-trial Python dispatch from the optimization hot path.
VQE_BATCH_SIZE = 20

# Energy-error thresholds used for the final-energy success probabilities.
# Divide the decade from 1e-1 to 1e-2 into ten equal logarithmic intervals,
# including both endpoints (eleven thresholds in total).
SUCCESS_PROBABILITY_THRESHOLDS = tuple(
    10.0 ** (-1.0 - division / 10.0)
    for division in range(11)
)

# Plot-specific energy-error thresholds for
# ``success_probability_multiple_tolerances.pdf``.  Keep them in strictly
# decreasing order because the success-count validator follows that order;
# the plotting routine presents them in increasing order in the legend.
SUCCESS_PROBABILITY_FIGURE_THRESHOLDS = tuple(
    multiple * 1e-1
    for multiple in range(10, 0, -1)
) + tuple(
    multiple * 1e-2
    for multiple in range(9, 0, -1)
)

# ------------------------------------------------------------
# Layer schedules
# ------------------------------------------------------------
# DPQC VQE optimization
VQE_MAX_LAYER = 44
VQE_DENSE_UNTIL_LAYER = 8
VQE_SPARSE_STEP = 4

# Unitary-PQC VQE optimization
UNITARY_PQC_MAX_LAYER = 44
UNITARY_PQC_DENSE_UNTIL_LAYER = 8
UNITARY_PQC_SPARSE_STEP = 4

# DPQC random-point QFIM/derivative analyses
DPQC_QFIM_MAX_LAYER = 44
DPQC_QFIM_DENSE_UNTIL_LAYER = 8
DPQC_QFIM_SPARSE_STEP = 4

# Unitary-PQC random-point QFIM/derivative analyses
UNITARY_PQC_QFIM_MAX_LAYER = UNITARY_PQC_MAX_LAYER
UNITARY_PQC_QFIM_DENSE_UNTIL_LAYER = UNITARY_PQC_DENSE_UNTIL_LAYER
UNITARY_PQC_QFIM_SPARSE_STEP = UNITARY_PQC_SPARSE_STEP

# Backward-compatible aliases for the former DPQC-only QFIM names.
QFIM_MAX_LAYER = DPQC_QFIM_MAX_LAYER
QFIM_DENSE_UNTIL_LAYER = DPQC_QFIM_DENSE_UNTIL_LAYER
QFIM_SPARSE_STEP = DPQC_QFIM_SPARSE_STEP

# ------------------------------------------------------------
# QFIM numerical settings
# ------------------------------------------------------------
NUM_QFIM_SAMPLES = NUM_RUNS
QFIM_EFFECTIVE_RANK_THRESHOLD = 1e-12
EIG_SUM_EPS = 1e-12
QFIM_EIG_PLOT_EPS = 1e-16
QFIM_SAMPLE_SEED_BASE = 0
UNITARY_PQC_QFIM_SAMPLE_SEED_BASE = 123456
PURE_QFIM_LAYER_THRESHOLD = 8
RED_JVP_CHUNK = 16
# Independent parameter points used by QFIM, HS, and Hessian analyses
# are evaluated with a fixed-shape ``jit(vmap(...))`` runner.  Keep this
# smaller than VQE_BATCH_SIZE because derivative matrices grow quadratically
# with the number of circuit parameters.
ANALYSIS_BATCH_SIZE = 1

# Save trace-based / participation-rank QFIM diagnostics in addition to the
# legacy threshold-rank outputs.  These switches do not trigger independent
# QFIM evaluations: all metrics are derived from the same eigendecomposition.
RUN_QFIM_EFFECTIVE_RANK_RANDOM_POINTS = True
RUN_QFIM_EFFECTIVE_RANK_OPTIMIZATION_PATH = True

# Observable-Relevant Tangent Kernel (ORTK) numerical settings.
# Unitary-PQC does not require ORTK in the standard numerical pipeline.
RUN_UNITARY_PQC_ORTK_ANALYSIS = False
ORTK_RANK_THRESHOLD = 1e-12
PARTICIPATION_EFFECTIVE_RANK_EPS = 1e-30
ORTK_PARTICIPATION_EPS = PARTICIPATION_EFFECTIVE_RANK_EPS


# Thresholds used for QFIM eigenvalue-count plots along the optimization path.
QFIM_PATH_EIGCOUNT_THRESHOLDS = (1e1, 5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4)


# ------------------------------------------------------------
# Hamiltonian-observable invisible-tangent diagnostics
# ------------------------------------------------------------
RUN_INVISIBLE_TANGENT_RANDOM_POINTS = True
RUN_INVISIBLE_TANGENT_OPTIMIZATION_PATH = True

INVISIBLE_TANGENT_QFIM_THRESHOLDS = (
    1e-4,
    1e-6,
    1e-8,
    1e-10,
    1e-12,
    1e-14,
)
INVISIBLE_TANGENT_QFIM_RELATIVE_THRESHOLD = 0.0
INVISIBLE_TANGENT_OBS_SVD_ABS_THRESHOLD = 1e-12
INVISIBLE_TANGENT_OBS_SVD_REL_THRESHOLD = 1e-10

INVISIBLE_TANGENT_NORM_EPS = 1e-24
INVISIBLE_TANGENT_JVP_CHUNK = 16

# None uses all existing random-QFIM samples/layers.
INVISIBLE_TANGENT_RANDOM_NUM_SAMPLES = None
INVISIBLE_TANGENT_RANDOM_LAYERS = None

# None uses all available VQE layers/runs/sampled optimization iterations.
INVISIBLE_TANGENT_OPT_LAYERS = None
INVISIBLE_TANGENT_OPT_RUN_INDICES = None
INVISIBLE_TANGENT_OPT_TARGET_ITERATIONS = None

QFIM_DEGENERACY_RTOL = 1e-8
QFIM_DEGENERACY_ATOL = 1e-12
