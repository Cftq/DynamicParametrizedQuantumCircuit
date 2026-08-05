#!/usr/bin/env python
# coding: utf-8
"""Shared JAX precision contract for DPQC and Unitary-PQC modules."""

from __future__ import annotations

import os

# JAX reads this switch while it is imported.  Set it first so every caller
# gets the same float64/complex128 contract, even in a fresh subprocess.
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp


def ensure_jax_x64() -> None:
    """Enable the float64/complex128 precision required by the analyses."""
    os.environ["JAX_ENABLE_X64"] = "1"
    jax.config.update("jax_enable_x64", True)
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("JAX x64 precision could not be enabled.")


ensure_jax_x64()

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128


__all__ = ["COMPLEX_DTYPE", "REAL_DTYPE", "ensure_jax_x64"]
