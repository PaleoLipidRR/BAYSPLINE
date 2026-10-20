"""SST from UK'37 — ``UK_predict.m`` (PORTING.md UKP-01..14)."""
from __future__ import annotations

import time

import numpy as np

from .convergence import rhat_from_moments
from .parallel import resolve_n_jobs, run_blocks
from .posterior import load_posterior
from .results import SSTPrediction
from .spline import Spline
from .utils import as_column, matlab_round, normpdf

#: Thinning of the stored posterior. UK_predict.m:66-67, ``(1:2:end)``.
POSTERIOR_THIN = 2
#: Metropolis-Hastings iterations per chain, and how many to discard. UK_predict.m:75-77.
N_ITER, BURNIN = 500, 250
#: Prahl et al. (1988) culture calibration used for the prior mean. UK_predict.m:80.
PRAHL_INTERCEPT, PRAHL_SLOPE = 0.039, 0.034
#: Stride through the pooled samples that makes ``output.ens``. UK_predict.m:174.
ENS_STRIDE = 125
#: Percentiles of ``output.SST``. UK_predict.m:177.
SST_LEVELS = (0.025, 0.50, 0.975)
#: Posterior draws per block of work. Fixed, so results do not depend on n_jobs.
CHAINS_PER_BLOCK = 10


def jump_width(prior_mean: np.ndarray) -> float:
    """Proposal SD from the median prior mean, for a ~44% acceptance (UKP-05).

    Args:
        prior_mean: (N,) prior means, °C.

    Returns:
        The jumping-distribution standard deviation, °C.
    """
    pmm = float(np.median(prior_mean))
    if pmm < 20:
        return 3.5
    if 20 <= pmm <= 23.7:
        return 3.7
    return pmm * 0.8092 - 15.1405


def _sample_block(uk, prior_mean, prior_sd, jw, coefs, tau2, seed_seq, knots,
                  n_iter, burnin):
    """Run every chain in one block of posterior draws (UKP-06, UKP-07).

    MATLAB's ``parfor jj`` body, with the ``jj`` loop moved inside the arrays:
    row ``p`` of every (n_block, N) array is chain ``jj`` of the original.

    Returns:
        ``(samples, n_accepted, chain_mean, chain_var)`` — samples are
        (N, n_block * (n_iter - burnin)), each point's retained states
        chain-major and contiguous, which is the layout the percentiles want;
        accepts are counted over the same iterations; the moments, (n_block, N),
        feed R-hat (UKP-10).
    """
    spline = Spline(knots)
    rng = np.random.default_rng(seed_seq)
    n_block, n = len(tau2), len(uk)
    sigma = np.sqrt(tau2)[:, None]
    s_now = np.broadcast_to(prior_mean, (n_block, n)).copy()
    s0_now = normpdf(uk, spline(s_now, coefs), sigma) * normpdf(s_now, prior_mean, prior_sd)
    kept = np.empty((n, n_block, n_iter - burnin))
    n_acc = 0
    for kk in range(1, n_iter):
        s_prop = s_now + jw * rng.standard_normal((n_block, n))
        s0_p = normpdf(uk, spline(s_prop, coefs), sigma) * normpdf(s_prop, prior_mean, prior_sd)
        # MATLAB's min ignores NaN, so a 0/0 ratio accepts; np.minimum would
        # propagate it and reject. fmin is MATLAB's min (UKP-07).
        with np.errstate(divide="ignore", invalid="ignore"):
            success = np.fmin(1.0, s0_p / s0_now)
        accept = rng.random((n_block, n)) <= success
        s_now = np.where(accept, s_prop, s_now)
        s0_now = np.where(accept, s0_p, s0_now)
        if kk >= burnin:
            kept[:, :, kk - burnin] = s_now.T
            n_acc += int(accept.sum())
    return (kept.reshape(n, -1), n_acc,
            kept.mean(axis=2).T, kept.var(axis=2, ddof=1).T)


def _order_statistics(pooled, kth, n_jobs):
    """``sort(MH_c, 2)`` then index, without the full sort (UKP-13).

    Only three order statistics are wanted, so :func:`numpy.partition` finds them
    in linear time. Rows are independent, and partition releases the GIL, so
    the rows are split over threads.
    """
    rows = np.array_split(np.arange(pooled.shape[0]), min(resolve_n_jobs(n_jobs), pooled.shape[0]))
    parts = run_blocks(lambda r: np.partition(pooled[r], kth, axis=1)[:, kth],
                       [(r,) for r in rows], n_jobs, backend="thread")
    return np.concatenate(parts)


def uk_predict(uk, pstd: float, bayes=None, *, seed=None, n_jobs: int | None = -1,
               backend: str = "process", ens: str = "matlab",
               save_samples: bool = False) -> SSTPrediction:
    """SST from UK'37 with the BAYSPLINE calibration (``UK_predict.m``).

    Metropolis-Hastings, one chain per retained posterior draw of the spline (500
    of them), 500 iterations with 250 discarded — MATLAB's sampler, step for
    step. The chains are independent, so they run on every CPU at once:
    MATLAB's ``parfor`` becomes `n_jobs`.

    Args:
        uk: (N,) UK'37 values.
        pstd: Prior standard deviation, °C. 5 or 10 are the documented choices.
        bayes: Another posterior file, as MATLAB's optional third argument.
        seed: Anything :class:`numpy.random.SeedSequence` accepts. The result
            depends on the seed and not on `n_jobs` (UKP-08).
        n_jobs: Workers. ``-1`` (the default) is every CPU available to this
            process, ``1`` runs in the calling thread.
        backend: ``"process"`` (default) or ``"thread"``; see
            :func:`~baysplinepy.parallel.run_blocks`.
        ens: ``"matlab"`` reproduces ``output.ens`` exactly — every 125th pooled
            sample, which is all 250 retained states of only four chains.
            ``"spread"`` takes two states from each of the 500 chains (UKP-12).
        save_samples: Keep the full (N, 125000) pooled sample on the result, in
            MATLAB's ``MH_c`` column order.

    Returns:
        An :class:`~baysplinepy.results.SSTPrediction`.

    Raises:
        ValueError: on non-finite UK'37, a non-positive `pstd`, or an unknown
            `ens`.
    """
    t0 = time.perf_counter()
    uk = as_column(uk, "uk")
    if uk.size == 0 or not np.all(np.isfinite(uk)):
        raise ValueError("uk must be a non-empty vector of finite UK'37 values (UKP-02)")
    pstd = float(pstd)
    if not np.isfinite(pstd) or pstd <= 0:
        raise ValueError(f"pstd must be a positive number of degrees C, got {pstd}")
    if ens not in ("matlab", "spread"):
        raise ValueError(f"ens must be 'matlab' or 'spread', got {ens!r}")

    post = load_posterior(bayes)
    bdraws = post.bdraws[::POSTERIOR_THIN]
    tau2 = post.tau2[::POSTERIOR_THIN]
    n_ts, n_p, n_keep = uk.size, tau2.size, N_ITER - BURNIN

    prior_mean = (uk - PRAHL_INTERCEPT) / PRAHL_SLOPE
    jw = jump_width(prior_mean)

    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    blocks = np.array_split(np.arange(n_p), -(-n_p // CHAINS_PER_BLOCK))
    children = ss.spawn(len(blocks))
    jobs = [(uk, prior_mean, pstd, jw, bdraws[b], tau2[b], child, post.knots, N_ITER, BURNIN)
            for b, child in zip(blocks, children)]
    results = run_blocks(_sample_block, jobs, n_jobs, backend)

    # Each point's 125,000 retained states, chain-major: column jj*n_keep + t is
    # chain jj at kept iteration t.
    pooled = np.concatenate([r[0] for r in results], axis=1)
    accepted = sum(r[1] for r in results)
    rhat = rhat_from_moments(np.concatenate([r[2] for r in results]),
                             np.concatenate([r[3] for r in results]), n_keep)

    # MATLAB pools with reshape(MH_samps_t, N_Ts, N_p*(N-burnin)), which is
    # column-major: its column c is chain c mod N_p at kept iteration c div N_p.
    # Only ens and the saved samples depend on that order (UKP-11).
    c = np.arange(0, n_p * n_keep, ENS_STRIDE)
    if ens == "matlab":
        ensemble = pooled[:, (c % n_p) * n_keep + c // n_p]
    else:
        ensemble = pooled[:, c]
    kth = matlab_round(np.asarray(SST_LEVELS) * pooled.shape[1]).astype(int) - 1
    sst = _order_statistics(pooled, kth, n_jobs)
    mh_c = None
    if save_samples:
        mh_c = pooled.reshape(n_ts, n_p, n_keep).transpose(0, 2, 1).reshape(n_ts, -1)

    return SSTPrediction(
        sst=sst, ensemble=ensemble, prior_mean=prior_mean,
        prior_std=pstd, jump_dist=jw, rhat=float(np.median(rhat)),
        accepts=accepted / (n_ts * n_p * n_keep),
        samples=mh_c,
        metadata={"n_chains": n_p, "n_iter": N_ITER, "burnin": BURNIN,
                  "rhat_per_point": rhat, "ens": ens, "seed_entropy": ss.entropy,
                  "n_jobs": min(resolve_n_jobs(n_jobs), len(blocks)), "backend": backend,
                  "posterior": post.source, "seconds": time.perf_counter() - t0},
    )
