"""Against the original MATLAB functions, executing.

``docs/BAYSPLINEpy/audit/verify_original.m`` runs ``UK_predict.m``,
``UK_forward.m`` and ``checkSeasonality.m`` unmodified and saves what they
return. Deterministic quantities must agree exactly; Monte-Carlo ones are
compared against the spread of this port across seeds.

How the stochastic comparison is judged: the reference is one MATLAB run, so
its percentile for a point differs from the true value by one run's noise. The
port is run K times; the mean of those has noise sd/sqrt(K). The difference is
judged against sd*sqrt(1 + 1/K) at 5 sigma, per cell. The run-to-run variance
is pooled over points within each column, because eight runs alone estimate a
per-cell sd too loosely for a 5-sigma bound to mean much (a t with 7 degrees of
freedom exceeds 5 one time in 600).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.io as sio

import baysplinepy as bp

K = 8
Z = 5.0


@pytest.fixture(scope="module")
def ref(root):
    path = root / "docs" / "BAYSPLINEpy" / "audit" / "original_matlab.mat"
    if not path.is_file():
        pytest.skip("run docs/BAYSPLINEpy/audit/verify_original.m first")
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _runs(uk, pstd):
    return [bp.uk_predict(uk, pstd, seed=s) for s in range(K)]


@pytest.fixture(scope="module")
def demo_runs(ref):
    inp = ref["demo_input"]
    return _runs(np.atleast_1d(inp.uk), float(inp.pstd))


@pytest.fixture(scope="module")
def low_runs(ref):
    inp = ref["low_input"]
    return _runs(np.atleast_1d(inp.uk), float(inp.pstd))


def _judge(ref_vals, runs_vals):
    """z-scores of the reference against K runs; variance pooled over axis 0."""
    runs_vals = np.asarray(runs_vals, dtype=float)
    mean = runs_vals.mean(0)
    var = runs_vals.var(0, ddof=1)
    if var.ndim:
        var = np.broadcast_to(var.mean(axis=0), var.shape)
    return (np.asarray(ref_vals) - mean) / np.sqrt(var * (1 + 1 / K))


@pytest.mark.parametrize("case", ["demo", "low"])
def test_prior_mean_and_jump_width_exact(ref, case, demo_runs, low_runs):
    runs = demo_runs if case == "demo" else low_runs
    out = ref[case]
    np.testing.assert_allclose(runs[0].prior_mean, np.atleast_1d(out.prior_mean), rtol=1e-13)
    assert runs[0].prior_std == float(out.prior_std)
    assert runs[0].jump_dist == pytest.approx(float(out.jump_dist), abs=1e-12)


@pytest.mark.parametrize("case", ["demo", "low"])
def test_shapes_as_matlab(ref, case, demo_runs, low_runs):
    runs = demo_runs if case == "demo" else low_runs
    out = ref[case]
    assert runs[0].sst.shape == np.atleast_2d(out.SST).shape
    assert runs[0].ensemble.shape == np.atleast_2d(out.ens).shape


@pytest.mark.parametrize("case", ["demo", "low"])
def test_sst_percentiles_within_sampling_noise(ref, case, demo_runs, low_runs):
    runs = demo_runs if case == "demo" else low_runs
    z = _judge(np.atleast_2d(ref[case].SST), [r.sst for r in runs])
    worst = np.unravel_index(np.abs(z).argmax(), z.shape)
    assert np.abs(z).max() < Z, f"cell {worst}: z = {z[worst]:.2f}"
    # and no systematic offset: the mean z is near zero for an unbiased port
    assert abs(z.mean()) < Z / np.sqrt(z.size) * 3


@pytest.mark.parametrize("case", ["demo", "low"])
def test_acceptance_and_rhat_within_sampling_noise(ref, case, demo_runs, low_runs):
    runs = demo_runs if case == "demo" else low_runs
    for field, attr in (("accepts", "accepts"), ("rhat", "rhat")):
        z = _judge(float(getattr(ref[case], field)), [getattr(r, attr) for r in runs])
        assert abs(z) < Z, f"{field}: z = {z:.2f}"


def test_ensemble_distribution(ref, demo_runs):
    """output.ens: same per-point mean and spread, within noise (four chains each)."""
    m_ens = np.atleast_2d(ref["demo"].ens)
    for stat in (np.mean, np.std):
        z = _judge(stat(m_ens, axis=1), [stat(r.ensemble, axis=1) for r in demo_runs])
        assert np.abs(z).max() < Z


def test_spline_exact(ref):
    """fnval(fnxtr(spmak(augknt(knots,3), bdraws)), x) — no randomness at all."""
    p = bp.load_posterior()
    x = np.atleast_1d(ref["spline_x"]).astype(float)
    np.testing.assert_allclose(bp.Spline(p.knots)(x, p.bdraws), ref["spline_mean"],
                               rtol=0, atol=1e-12)


def test_forward_distribution(ref):
    m = np.atleast_2d(ref["forward"])
    ssts = np.atleast_1d(ref["forward_input"].ssts).astype(float)
    assert m.shape == (ssts.size, 1000)
    assert m.min() >= 0 and m.max() <= 1
    runs = [bp.uk_forward(ssts, seed=s).uk for s in range(K)]
    for stat in (np.mean, np.std):
        z = _judge(stat(m, axis=1), [stat(r, axis=1) for r in runs])
        assert np.abs(z).max() < Z


def test_seasonality_exact(ref):
    months = bp.check_seasonality(ref["season_lat"], ref["season_lon"])
    mask = np.zeros((len(months), 12), dtype=bool)
    for i, m in enumerate(months):
        mask[i, m - 1] = True
    np.testing.assert_array_equal(mask, np.atleast_2d(ref["season_months"]).astype(bool))
