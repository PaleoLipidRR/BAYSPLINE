"""UK'37 from SST — ``UK_forward.m`` (PORTING.md UKF-01..04)."""
from __future__ import annotations

import numpy as np

from .posterior import load_posterior
from .results import UKPrediction
from .spline import Spline
from .utils import as_column


def uk_forward(ssts, bayes=None, *, seed=None, clip: bool = True) -> UKPrediction:
    """Forward-model UK'37 from SST (``UK_forward.m``).

    Every posterior draw is used — 1000 from the default file, not thinned —
    giving one simulated UK'37 per draw per temperature.

    The calibration is seasonal in the North Atlantic, North Pacific and
    Mediterranean; pass the matching seasonal SSTs there. See
    :func:`~baysplinepy.seasonality.check_seasonality`.

    Args:
        ssts: (N,) sea-surface temperatures, °C.
        bayes: Another posterior file, as MATLAB's optional second argument.
        seed: Anything :func:`numpy.random.default_rng` accepts.
        clip: Force values into [0, 1], the bounds of the index, as MATLAB
            does (the default). ``False`` keeps the unclipped draws.

    Returns:
        A :class:`~baysplinepy.results.UKPrediction`, ``uk`` of shape
        (N, n_draws).
    """
    ssts = as_column(ssts, "ssts")
    post = load_posterior(bayes)
    rng = np.random.default_rng(seed)
    # fnval of the vector-valued spline: (n_draws, N), then transposed (UKF-02).
    mean_now = Spline(post.knots)(ssts, post.bdraws)
    uk = (mean_now + np.sqrt(post.tau2)[:, None] * rng.standard_normal(mean_now.shape)).T
    clipped = float(np.mean((uk > 1) | (uk < 0)))
    if clip:
        uk = np.clip(uk, 0.0, 1.0)
    return UKPrediction(uk=uk, metadata={"clipped_fraction": clipped, "clip": clip,
                                         "n_draws": post.n_draws, "posterior": post.source})
