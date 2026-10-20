"""Result containers — the Python form of MATLAB's ``output`` structure."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SSTPrediction:
    """SSTs inferred from UK'37 (``UK_predict.m``'s ``output``; PORTING.md UKP-13).

    Attributes:
        sst: (N, 3) the 2.5th, 50th and 97.5th percentiles, °C (``output.SST``).
        ensemble: (N, 1000) posterior SSTs (``output.ens``).
        prior_mean: (N,) prior means from the Prahl et al. (1988) equation.
        prior_std: Prior standard deviation, as given.
        jump_dist: Proposal standard deviation used.
        rhat: Median R-hat across points.
        accepts: Acceptance rate over the retained iterations.
        samples: (N, 125000) every retained state, if `save_samples` was set.
        metadata: Provenance — chains, iterations, seed, workers, timing.
    """

    sst: np.ndarray
    ensemble: np.ndarray
    prior_mean: np.ndarray
    prior_std: float
    jump_dist: float
    rhat: float
    accepts: float
    samples: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def p2_5(self) -> np.ndarray:
        """The 2.5th percentile, °C."""
        return self.sst[:, 0]

    @property
    def p50(self) -> np.ndarray:
        """The median, °C."""
        return self.sst[:, 1]

    @property
    def p97_5(self) -> np.ndarray:
        """The 97.5th percentile, °C."""
        return self.sst[:, 2]

    def percentile(self, q=(2.5, 50, 97.5), method: str = "matlab") -> np.ndarray:
        """Percentiles of the posterior.

        Args:
            q: Percentiles in 0-100.
            method: ``"matlab"`` sorts every retained sample and indexes it as
                ``UK_predict.m`` does; anything else is passed to
                :func:`numpy.percentile` as `method`, over :attr:`samples` when
                kept and :attr:`ensemble` otherwise.

        Returns:
            (N, len(q)) array.
        """
        if method == "matlab":
            from .utils import matlab_round
            pool = self.samples if self.samples is not None else self.ensemble
            kth = matlab_round(np.asarray(q, dtype=float) / 100 * pool.shape[1]).astype(int) - 1
            kth = np.clip(kth, 0, pool.shape[1] - 1)
            return np.partition(pool, np.unique(kth), axis=1)[:, kth]
        pool = self.samples if self.samples is not None else self.ensemble
        return np.percentile(pool, q, axis=1, method=method).T

    def to_dict(self) -> dict[str, Any]:
        """MATLAB's field names, for diffing against a saved ``output``.

        Returns:
            A dict keyed as MATLAB's structure is.
        """
        return {"prior_mean": self.prior_mean, "prior_std": self.prior_std,
                "jump_dist": self.jump_dist, "rhat": self.rhat,
                "accepts": self.accepts, "ens": self.ensemble, "SST": self.sst}

    def to_dataframe(self):
        """The percentiles and prior mean as a DataFrame.

        Returns:
            A pandas DataFrame with columns ``SST_2_5``, ``SST_50``, ``SST_97_5``
            and ``prior_mean`` — the column names ``DemoScript.m`` writes.

        Raises:
            ImportError: if pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_dataframe() needs pandas: pip install pandas") from exc
        return pd.DataFrame({"SST_2_5": self.sst[:, 0], "SST_50": self.sst[:, 1],
                             "SST_97_5": self.sst[:, 2], "prior_mean": self.prior_mean})


@dataclass
class UKPrediction:
    """UK'37 forward-modelled from SST (``UK_forward.m``; PORTING.md UKF-04).

    Attributes:
        uk: (N, n_draws) simulated UK'37, clipped to [0, 1].
        metadata: Provenance, including ``clipped_fraction``.
    """

    uk: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ensemble(self) -> np.ndarray:
        """Alias of :attr:`uk`, the name ``brews/baysplinepy`` uses."""
        return self.uk

    def percentile(self, q=(2.5, 50, 97.5), method: str = "hazen") -> np.ndarray:
        """Percentiles across the ensemble.

        Args:
            q: Percentiles in 0-100.
            method: Passed to :func:`numpy.percentile`. ``"hazen"`` is MATLAB's
                ``prctile`` convention.

        Returns:
            (N, len(q)) array.
        """
        return np.percentile(self.uk, q, axis=1, method=method).T
