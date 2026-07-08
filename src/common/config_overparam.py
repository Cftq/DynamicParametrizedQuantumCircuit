#!/usr/bin/env python
# coding: utf-8
"""Configuration for DPQC overparameterization computations and plots."""


# ------------------------------------------------------------
# VQE optimization settings
# ------------------------------------------------------------
H_PARAM = 0.50
NUM_RUNS = 100
STEPS = 5000
SAMPLE_EVERY = 1000
LEARNING_RATE = 1e-3
TOLERANCE = 1e-2

# ------------------------------------------------------------
# Layer schedules
# ------------------------------------------------------------
VQE_MAX_LAYER = 5
VQE_DENSE_UNTIL_LAYER = 5
VQE_SPARSE_STEP = 5

UNITARY_PQC_MAX_LAYER = 2
UNITARY_PQC_DENSE_UNTIL_LAYER = 7
UNITARY_PQC_SPARSE_STEP = 1

# QFIM layer schedules
QFIM_MAX_LAYER = 40
QFIM_DENSE_UNTIL_LAYER = 8
QFIM_SPARSE_STEP = 4

# ------------------------------------------------------------
# QFIM numerical settings
# ------------------------------------------------------------
NUM_QFIM_SAMPLES = 100
QFIM_EFFECTIVE_RANK_THRESHOLD = 1e-12
EIG_SUM_EPS = 1e-12
QFIM_EIG_PLOT_EPS = 1e-16
QFIM_SAMPLE_SEED_BASE = 0
UNITARY_PQC_QFIM_SAMPLE_SEED_BASE = 123456
PURE_QFIM_LAYER_THRESHOLD = 8
RED_JVP_CHUNK = 16


# Thresholds used for large-sector gradient-weight diagnostics.
GRADIENT_SECTOR_THRESHOLDS = (1e1, 5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4)

# Thresholds used for QFIM eigenvalue-count plots along the optimization path.
QFIM_PATH_EIGCOUNT_THRESHOLDS = (1e1, 5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4)


# ------------------------------------------------------------
# QFIM-gradient alignment result generation / visualization
# ------------------------------------------------------------
RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER = False
RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES = False
RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION = True

LOG_X_QFIM_GRAD_ALIGNMENT = True
LOG_Y_QFIM_GRAD_ALIGNMENT = False
QFIM_GRAD_ALIGNMENT_RUN_INDICES = None

# None means all iterations in sample_iters:
#   1, SAMPLE_EVERY, 2*SAMPLE_EVERY, ..., STEPS
QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS = None
