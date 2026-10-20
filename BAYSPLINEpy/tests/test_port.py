"""One test per PORTING.md entry. The test name carries the entry ID."""
from __future__ import annotations

import numpy as np
import pytest

import baysplinepy as bp
from baysplinepy.uk_predict import (BURNIN, ENS_STRIDE, N_ITER, POSTERIOR_THIN, _sample_block,
                                    jump_width)
from baysplinepy.utils import matlab_round, normpdf


# --- data -------------------------------------------------------------------

def test_STO_01(root):
    """The packaged posterior is byte-identical to the MATLAB repository's."""
    from baysplinepy.posterior import DEFAULT_FILE
    assert DEFAULT_FILE.read_bytes() == (root / "bayes_posterior_v2.mat").read_bytes()
    from baysplinepy.demo import DEMO_FILE
    assert DEMO_FILE.read_bytes() == (root / "DemoUKData.csv").read_bytes()


def test_STO_02(tmp_path, posterior):
    """``bayes`` takes another file, as MATLAB's optional argument does."""
    import scipy.io as sio
    alt = tmp_path / "alt.mat"
    sio.savemat(alt, {"bdraws": posterior.bdraws[:10], "tau2": posterior.tau2[:10, None],
                      "knots": posterior.knots[None]})
    p = bp.load_posterior(alt)
    assert p.n_draws == 10
    out = bp.uk_predict([0.5, 0.6], 5, bayes=alt, seed=0, n_jobs=1)
    assert out.metadata["n_chains"] == 5            # 10 draws, thinned 1:2:end
    with pytest.raises(KeyError, match="bdraws"):
        sio.savemat(tmp_path / "bad.mat", {"tau2": [1.0]})
        bp.load_posterior(tmp_path / "bad.mat")


# --- the spline -------------------------------------------------------------

def test_SPL_01():
    """augknt: end knots to multiplicity exactly k, duplicates dropped."""
    np.testing.assert_array_equal(bp.augknt([-0.4, 23.6, 29.6], 3),
                                  [-0.4, -0.4, -0.4, 23.6, 29.6, 29.6, 29.6])
    np.testing.assert_array_equal(bp.augknt([0, 0, 1, 2], 2), [0, 0, 1, 2, 2])


def test_SPL_02(posterior):
    """Inside the knots the spline is the B-spline, scipy's evaluation."""
    from scipy.interpolate import BSpline
    s = bp.Spline(posterior.knots)
    x = np.linspace(-0.4, 29.6, 301)
    for c in posterior.bdraws[:20]:
        ref = BSpline(s.t, c, 2)(x)
        np.testing.assert_allclose(s(x, c), ref, rtol=0, atol=1e-14)


def test_SPL_03(posterior):
    """fnxtr: linear beyond the knots, matching value and slope at the end."""
    s = bp.Spline(posterior.knots)
    c = posterior.bdraws[3]
    for end, sign in ((-0.4, -1), (29.6, 1)):
        h = 1e-6
        slope_in = (s(end, c) - s(end - sign * h, c)) / h * sign
        far = s(end + sign * 10, c)
        np.testing.assert_allclose((far - s(end, c)) / (10 * sign), slope_in, rtol=1e-5)
        # and straight: second differences vanish outside
        x = end + sign * np.array([1.0, 2.0, 3.0])
        y = s(x, c)
        assert abs(y[0] - 2 * y[1] + y[2]) < 1e-12


def test_SPL_04(posterior):
    """The many-splines fast path agrees with evaluating each spline alone."""
    s = bp.Spline(posterior.knots)
    rng = np.random.default_rng(1)
    x = rng.normal(15, 15, (40, 30))
    c = posterior.bdraws[:40]
    one_by_one = np.array([s(x[i], c[i]) for i in range(40)])
    np.testing.assert_allclose(s(x, c), one_by_one, rtol=0, atol=1e-13)
    # shared x: fnval's d-by-N layout for a vector-valued spline
    assert s(x[0], c).shape == (40, 30)
    np.testing.assert_allclose(s(x[0], c)[5], s(x[0], c[5]), atol=1e-13)


def test_SPL_05(posterior):
    """Independent check: brews/baysplinepy's own fnxtr port gives the same curve."""
    butils = pytest.importorskip("bayspline.utils", reason="brews/baysplinepy not installed")
    s = bp.Spline(posterior.knots)
    x = np.linspace(-30, 60, 2001)
    for c in posterior.bdraws[:25]:
        theirs = butils.extrapolate_spline([butils.augknt(posterior.knots, 2), c, 2])(x)
        np.testing.assert_allclose(s(x, c), theirs, rtol=0, atol=1e-13)


# --- UK_predict.m -----------------------------------------------------------

def test_UKP_01(posterior):
    """Draws are thinned 1:2:end — 500 chains from the 1000 stored."""
    assert POSTERIOR_THIN == 2
    out = bp.uk_predict([0.5], 5, seed=0, n_jobs=1)
    assert out.metadata["n_chains"] == len(posterior.tau2[::2]) == 500


def test_UKP_02():
    """Input checks MATLAB lacks: NaN UK'37 would make every proposal NaN."""
    with pytest.raises(ValueError, match="finite"):
        bp.uk_predict([0.5, np.nan], 5)
    with pytest.raises(ValueError, match="pstd"):
        bp.uk_predict([0.5], 0)
    with pytest.raises(ValueError, match="flattens"):
        bp.uk_predict(np.ones((2, 2)) * 0.5, 5)
    # a row or a column is fine, as uk(:) makes either a column
    a = bp.uk_predict(np.array([[0.5, 0.6]]), 5, seed=0, n_jobs=1)
    b = bp.uk_predict([0.5, 0.6], 5, seed=0, n_jobs=1)
    np.testing.assert_array_equal(a.sst, b.sst)


def test_UKP_03():
    """Prior mean: Prahl et al. (1988), (uk - 0.039) / 0.034."""
    uk = np.array([0.1, 0.5, 0.9])
    out = bp.uk_predict(uk, 5, seed=0, n_jobs=1)
    np.testing.assert_array_equal(out.prior_mean, (uk - 0.039) / 0.034)
    assert out.prior_std == 5.0


def test_UKP_04():
    """500 iterations, the first 250 discarded."""
    assert (N_ITER, BURNIN) == (500, 250)
    out = bp.uk_predict([0.5], 5, seed=0, n_jobs=1, save_samples=True)
    assert out.samples.shape == (1, 500 * 250)


@pytest.mark.parametrize("pmm,jw", [(10.0, 3.5), (19.999, 3.5), (20.0, 3.7), (23.7, 3.7),
                                    (23.71, 23.71 * 0.8092 - 15.1405), (28.0, 28 * 0.8092 - 15.1405)])
def test_UKP_05(pmm, jw):
    """Jump width from the median prior mean, all three branches and both edges."""
    assert jump_width(np.array([pmm - 1, pmm, pmm + 1])) == pytest.approx(jw, abs=1e-12)


def test_UKP_06(posterior):
    """The chain step: start at the prior mean, propose N(s, JW), accept on rand <= ratio."""
    uk = np.array([0.3, 0.8])
    pm = (uk - 0.039) / 0.034
    ss = np.random.SeedSequence(5)
    coefs, tau2 = posterior.bdraws[:3], posterior.tau2[:3]
    kept, n_acc, _, _ = _sample_block(uk, pm, 5.0, 3.5, coefs, tau2, ss, posterior.knots, 6, 2)
    # replay the same stream by hand, chain by chain, as MATLAB's loop does
    rng = np.random.default_rng(ss)
    s = bp.Spline(posterior.knots)
    s_now = np.tile(pm, (3, 1))
    f = lambda x: normpdf(uk, s(x, coefs), np.sqrt(tau2)[:, None]) * normpdf(x, pm, 5.0)
    s0 = f(s_now)
    hand, acc = [], 0
    for kk in range(1, 6):
        prop = s_now + 3.5 * rng.standard_normal((3, 2))
        s0p = f(prop)
        a = rng.random((3, 2)) <= np.fmin(1, s0p / s0)
        s_now, s0 = np.where(a, prop, s_now), np.where(a, s0p, s0)
        if kk >= 2:
            hand.append(s_now.copy())
            acc += a.sum()
    hand = np.stack(hand, axis=-1)                     # (chains, N, kept)
    np.testing.assert_array_equal(kept.reshape(2, 3, 4), hand.transpose(1, 0, 2))
    assert n_acc == acc


def test_UKP_07():
    """MATLAB's min(1, NaN) is 1; np.fmin reproduces it, np.minimum would not."""
    assert np.fmin(1.0, np.nan) == 1.0
    assert np.isnan(np.minimum(1.0, np.nan))
    with np.errstate(invalid="ignore"):
        ratio = np.array([0.0]) / np.array([0.0])
    assert (np.random.default_rng(0).random(1) <= np.fmin(1, ratio)).all()


def test_UKP_08(demo_uk):
    """The answer depends on the seed, never on how many CPUs ran it."""
    one = bp.uk_predict(demo_uk[:20], 5, seed=11, n_jobs=1)
    two = bp.uk_predict(demo_uk[:20], 5, seed=11, n_jobs=2)
    allc = bp.uk_predict(demo_uk[:20], 5, seed=11, n_jobs=-1)
    thr = bp.uk_predict(demo_uk[:20], 5, seed=11, n_jobs=3, backend="thread")
    for other in (two, allc, thr):
        np.testing.assert_array_equal(one.sst, other.sst)
        np.testing.assert_array_equal(one.ensemble, other.ensemble)
        assert one.rhat == other.rhat and one.accepts == other.accepts
    diff = bp.uk_predict(demo_uk[:20], 5, seed=12, n_jobs=1)
    assert not np.array_equal(one.sst, diff.sst)


def test_UKP_09():
    """n_jobs follows joblib: -1 all CPUs, -2 all but one; 0 is an error."""
    from baysplinepy.parallel import available_cpus, resolve_n_jobs
    n = available_cpus()
    assert resolve_n_jobs(-1) == n
    assert resolve_n_jobs(None) == n
    assert resolve_n_jobs(-2) == max(1, n - 1)
    assert resolve_n_jobs(3) == 3
    with pytest.raises(ValueError):
        resolve_n_jobs(0)


def test_UKP_10(demo_uk):
    """R-hat: ChainConvergence's arithmetic, computed from worker moments."""
    out = bp.uk_predict(demo_uk[:6], 5, seed=1, n_jobs=1, save_samples=True)
    s = out.samples.T.reshape(250, 500, 6)              # MH_c columns are t*N_p + jj
    # ChainConvergence.m, literally, for point 0
    chains = s[:, :, 0]
    nc, m = chains.shape
    psi_j = chains.mean(0)
    bp_ = nc / (m - 1) * ((psi_j - psi_j.mean()) ** 2).sum()
    s2 = ((chains - np.kron(psi_j, np.ones((nc, 1)))) ** 2).sum(0) / (nc - 1)
    w = s2.mean()
    rhat0 = np.sqrt((((nc - 1) / nc) * w + bp_ / nc) / w)
    assert out.metadata["rhat_per_point"][0] == pytest.approx(rhat0, rel=1e-12)
    assert out.rhat == pytest.approx(np.median(bp.chain_convergence(s)), rel=1e-12)


def test_UKP_11(demo_uk):
    """MH_c is MATLAB's column-major reshape: column c is chain c mod 500, iteration c div 500."""
    out = bp.uk_predict(demo_uk[:4], 5, seed=2, n_jobs=1, save_samples=True)
    s = out.samples.T.reshape(250, 500, 4)
    c = np.arange(out.samples.shape[1])
    np.testing.assert_array_equal(out.samples[1], s[c // 500, c % 500, 1])


def test_UKP_12(demo_uk):
    """ens: every 125th pooled column — in MATLAB's order that is four chains only."""
    out = bp.uk_predict(demo_uk[:4], 5, seed=3, n_jobs=1, save_samples=True)
    assert ENS_STRIDE == 125
    np.testing.assert_array_equal(out.ensemble, out.samples[:, ::125])
    chains = (np.arange(0, 125000, 125) % 500)
    assert sorted(set(chains)) == [0, 125, 250, 375]
    spread = bp.uk_predict(demo_uk[:4], 5, seed=3, n_jobs=1, ens="spread")
    assert spread.ensemble.shape == (4, 1000)
    np.testing.assert_array_equal(spread.sst, out.sst)        # SST is unaffected
    # spread draws come from all 500 chains: two per chain
    s = out.samples.T.reshape(250, 500, 4)
    np.testing.assert_array_equal(spread.ensemble[0, ::2], s[0, :, 0])
    np.testing.assert_array_equal(spread.ensemble[0, 1::2], s[125, :, 0])


def test_UKP_13(demo_uk):
    """SST: sort the pooled sample and index round([.025 .5 .975] * n)."""
    out = bp.uk_predict(demo_uk[:5], 5, seed=4, n_jobs=1, save_samples=True)
    n = out.samples.shape[1]
    idx = matlab_round(np.array([0.025, 0.5, 0.975]) * n).astype(int) - 1
    np.testing.assert_array_equal(out.sst, np.sort(out.samples, axis=1)[:, idx])
    np.testing.assert_array_equal(out.percentile((2.5, 50, 97.5)), out.sst)
    assert set(out.to_dict()) == {"prior_mean", "prior_std", "jump_dist", "rhat",
                                  "accepts", "ens", "SST"}


def test_UKP_14(demo_uk):
    """accepts: the share of retained iterations whose proposal was accepted."""
    out = bp.uk_predict(demo_uk, 5, seed=0)
    # Tierney & Tingley tune JW for roughly 0.44
    assert 0.3 < out.accepts < 0.55


# --- UK_forward.m -----------------------------------------------------------

def test_UKF_01(posterior):
    """All 1000 draws, unthinned; output N x 1000."""
    out = bp.uk_forward([10.0, 20.0, 30.0], seed=0)
    assert out.uk.shape == (3, posterior.n_draws)


def test_UKF_02(posterior):
    """mean = fnval(bs, sst) per draw, noise sqrt(tau2) per draw."""
    out = bp.uk_forward([5.0, 25.0], seed=7, clip=False)
    rng = np.random.default_rng(7)
    s = bp.Spline(posterior.knots)
    mean = s(np.array([5.0, 25.0]), posterior.bdraws)
    ref = (mean + np.sqrt(posterior.tau2)[:, None] * rng.standard_normal(mean.shape)).T
    np.testing.assert_array_equal(out.uk, ref)


def test_UKF_03():
    """Clipped to [0, 1], with the clipped share recorded."""
    out = bp.uk_forward([-5.0, 35.0], seed=0)
    assert out.uk.min() >= 0 and out.uk.max() <= 1
    assert out.metadata["clipped_fraction"] > 0.2
    raw = bp.uk_forward([-5.0, 35.0], seed=0, clip=False)
    assert raw.uk.max() > 1 or raw.uk.min() < 0


def test_UKF_04():
    """Round trip: forward then predict recovers the temperature, roughly."""
    truth = np.array([8.0, 16.0, 24.0])
    uk = np.median(bp.uk_forward(truth, seed=0).uk, axis=1)
    out = bp.uk_predict(uk, 10, seed=0)
    np.testing.assert_allclose(out.p50, truth, atol=1.5)


# --- checkSeasonality.m -----------------------------------------------------

def test_CS_01():
    """Months per region; later regions override earlier ones, as in MATLAB."""
    m = bp.check_seasonality([40, 60, 55, 0], [15, -20, -170, 0])
    np.testing.assert_array_equal(m[0], [1, 2, 3, 4, 5, 11, 12])
    np.testing.assert_array_equal(m[1], [8, 9, 10])
    np.testing.assert_array_equal(m[2], [6, 7, 8])          # -170 -> 190 in 0-360
    np.testing.assert_array_equal(m[3], np.arange(1, 13))


def test_CS_02():
    """inpolygon counts edges and vertices as inside."""
    sq = (np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))
    pts = np.array([[0.5, 0.5], [0, 0.5], [1, 1], [0.5, 0], [1.5, 0.5], [-1e-9, 0.5]])
    np.testing.assert_array_equal(bp.inpolygon(pts[:, 0], pts[:, 1], *sq),
                                  [True, True, True, True, False, False])


def test_CS_03():
    """Mismatched lat/lon is an error rather than MATLAB's indexing failure."""
    with pytest.raises(ValueError):
        bp.check_seasonality([1, 2], [1])


# --- general ----------------------------------------------------------------

def test_GEN_01():
    """x(:) of a matrix is rejected, not silently flattened in another order."""
    from baysplinepy.utils import as_column
    np.testing.assert_array_equal(as_column([[1, 2, 3]]), [1, 2, 3])
    with pytest.raises(ValueError):
        as_column(np.zeros((2, 3)))


def test_GEN_02():
    """MATLAB round is half away from zero."""
    np.testing.assert_array_equal(matlab_round([0.5, 1.5, 2.5, -0.5]), [1, 2, 3, -1])


def test_GEN_03():
    """normpdf is the Statistics Toolbox formula."""
    from scipy.stats import norm
    x = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(normpdf(x, 0.5, 1.7), norm.pdf(x, 0.5, 1.7), rtol=1e-14)
