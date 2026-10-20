"""``ChainConvergence`` — the R-hat subfunction of ``UK_predict.m`` (PORTING.md UKP-10)."""
from __future__ import annotations

import numpy as np


def chain_convergence(samples: np.ndarray) -> np.ndarray:
    """Gelman's R-hat, for every predicted point at once.

    MATLAB calls ``ChainConvergence(squeeze(MH_samps_t(i,:,:)), N_p)`` once per
    point; this is the same arithmetic vectorised over the points.

    Note what the "chains" are: each runs under a *different* posterior draw of
    the spline, so their stationary distributions differ and R-hat measures
    calibration uncertainty as well as mixing. Values a little above 1 are
    expected and are not, on their own, a convergence failure.

    Args:
        samples: (n_iter, n_chains, N) retained states.

    Returns:
        (N,) R-hat per point.
    """
    psi_j = samples.mean(axis=0)                       # psi_bar_dot_j, (M, N)
    s2_j = ((samples - psi_j) ** 2).sum(axis=0) / (samples.shape[0] - 1)
    return rhat_from_moments(psi_j, s2_j, samples.shape[0])


def rhat_from_moments(psi_j: np.ndarray, s2_j: np.ndarray, nc: int) -> np.ndarray:
    """R-hat from each chain's mean and variance.

    The sampler's workers compute these for their own chains, so the pooled
    (n_iter, n_chains, N) array never has to be swept a second time.

    Args:
        psi_j: (n_chains, N) chain means.
        s2_j: (n_chains, N) chain variances, ``ddof=1``.
        nc: Iterations per chain.

    Returns:
        (N,) R-hat per point.
    """
    m = psi_j.shape[0]
    psi = psi_j.mean(axis=0)                           # psi_bar_dot_dot
    b = (nc / (m - 1)) * ((psi_j - psi) ** 2).sum(axis=0)
    w = s2_j.mean(axis=0)
    var_hat = ((nc - 1) / nc) * w + b / nc
    return np.sqrt(var_hat / w)


def effective_size(samples: np.ndarray) -> np.ndarray:
    """``Neff``, the second output of ``ChainConvergence`` (unused by MATLAB).

    Args:
        samples: (n_iter, n_chains, N) retained states.

    Returns:
        (N,) effective sample size.
    """
    nc, m = samples.shape[0], samples.shape[1]
    psi_j = samples.mean(axis=0)
    b = (nc / (m - 1)) * ((psi_j - psi_j.mean(axis=0)) ** 2).sum(axis=0)
    w = (((samples - psi_j) ** 2).sum(axis=0) / (nc - 1)).mean(axis=0)
    var_hat = ((nc - 1) / nc) * w + b / nc
    return m * nc * np.minimum(var_hat / b, 1.0)
