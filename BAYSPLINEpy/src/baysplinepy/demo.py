"""The demo record shipped with the MATLAB package."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DEMO_FILE = Path(__file__).with_name("data") / "DemoUKData.csv"


def load_demo_data() -> dict[str, np.ndarray]:
    """``readtable('DemoUKData.csv')`` — UK'37 from the Gulf of Aden.

    The file starts with a UTF-8 byte-order mark, which MATLAB's ``readtable``
    strips and a plain ``open`` does not.

    Returns:
        ``{"depth", "ageBP", "uk37"}``, each a 1-D float array.
    """
    with open(DEMO_FILE, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
