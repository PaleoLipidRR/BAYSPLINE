"""The fitted posterior, ``bayes_posterior_v2.mat`` (PORTING.md STO-01, STO-02)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

#: The file both MATLAB functions load when no ``bayes`` argument is given.
DEFAULT_FILE = Path(__file__).with_name("data") / "bayes_posterior_v2.mat"


@dataclass(frozen=True)
class Posterior:
    """Posterior draws of the spline coefficients and the residual variance.

    Attributes:
        bdraws: (n_draws, n_coefs) B-spline coefficients.
        tau2: (n_draws,) residual variance, UK'37 units squared.
        knots: Spline breakpoints, before augmentation.
        source: Where the draws were read from.
    """

    bdraws: np.ndarray
    tau2: np.ndarray
    knots: np.ndarray
    source: str = ""

    @property
    def n_draws(self) -> int:
        """Number of posterior draws."""
        return len(self.tau2)


def load_posterior(bayes=None) -> Posterior:
    """MATLAB ``load(bayes)`` / ``load('bayes_posterior_v2.mat')``.

    The default file ships inside the package; ``BAYSPLINE_POSTERIOR`` overrides
    it. Its bytes are tested equal to the copy in the MATLAB repository.

    Args:
        bayes: ``None`` for the default, a path to another ``.mat`` file with
            ``bdraws``, ``tau2`` and ``knots``, or a :class:`Posterior`.

    Returns:
        A :class:`Posterior`.
    """
    if isinstance(bayes, Posterior):
        return bayes
    if bayes is None:
        bayes = os.environ.get("BAYSPLINE_POSTERIOR", DEFAULT_FILE)
    return _load(str(Path(bayes).resolve()))


@lru_cache(maxsize=4)
def _load(path: str) -> Posterior:
    d = sio.loadmat(path)
    missing = {"bdraws", "tau2", "knots"} - set(d)
    if missing:
        raise KeyError(f"{path} lacks {sorted(missing)}; BAYSPLINE needs bdraws, tau2 and knots")
    bdraws = np.array(d["bdraws"], dtype=float)
    tau2 = np.array(d["tau2"], dtype=float).ravel()
    bdraws.flags.writeable = tau2.flags.writeable = False
    return Posterior(bdraws=bdraws, tau2=tau2,
                     knots=np.array(d["knots"], dtype=float).ravel(), source=path)
