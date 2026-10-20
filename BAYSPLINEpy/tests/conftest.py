"""Shared fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

import baysplinepy as bp


def repo_root() -> Path | None:
    """The MATLAB repository this package sits in, if it does."""
    for p in Path(__file__).resolve().parents:
        if (p / "UK_predict.m").is_file():
            return p
    return None


@pytest.fixture(scope="session")
def root():
    r = repo_root()
    if r is None:
        pytest.skip("not inside the MATLAB BAYSPLINE repository")
    return r


@pytest.fixture(scope="session")
def demo_uk():
    """The DemoScript.m record: UK'37 from the Gulf of Aden."""
    return bp.load_demo_data()["uk37"]


@pytest.fixture(scope="session")
def posterior():
    return bp.load_posterior()
