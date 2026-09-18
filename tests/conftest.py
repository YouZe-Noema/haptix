"""Shared pytest fixtures for the haptix test suite."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

# HAPTIX_TEST_SEED (default 0): seeds stdlib random and numpy.random so the
# suite is byte-for-byte reproducible. Override to exercise a different draw:
#   HAPTIX_TEST_SEED=3 python -m pytest -q


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    seed = int(os.environ.get("HAPTIX_TEST_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
