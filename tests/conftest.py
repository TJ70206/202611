from __future__ import annotations

import pytest

from cp202611.optimization.mvp_model import solve_mvp
from cp202611.synthetic import create_synthetic_mvp


@pytest.fixture(scope="session")
def synthetic_scenario():
    return create_synthetic_mvp()


@pytest.fixture(scope="session")
def solved_mvp(synthetic_scenario):
    return solve_mvp(synthetic_scenario)
