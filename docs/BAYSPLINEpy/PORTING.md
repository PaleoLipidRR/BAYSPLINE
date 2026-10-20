# MATLAB → Python translation record

This is the document to review. It walks every line of the three MATLAB functions —
`UK_predict.m`, `UK_forward.m`, `checkSeasonality.m` — and states what the Python does instead, so
a reviewer can check the recode **without reading the Python source**, and can disagree with a
specific numbered decision rather than with the port as a whole.

It follows the layout of `PaleoLipidRR/BAYSPAR`'s `docs/BAYSPARpy/PORTING.md`.

**The code these entries describe is in [`../../BAYSPLINEpy/`](../../BAYSPLINEpy/).** Run its
tests with `cd BAYSPLINEpy && PYTHONPATH=src python -m pytest`.

---

## How to review this

**Every entry has an ID** (`UKP-07`, `SPL-03`, …). Quote the ID when you disagree. IDs are stable.

**Every entry has a status:**

| Status | Meaning | Needs review? |
|---|---|---|
| `EXACT` | Same arithmetic in the same order. | No — check the translation is what you'd write. |
| `EQUIV` | Different expression, same result, pinned by a test. | Skim. |
| `EXTENSION` | New capability. Defaults reproduce MATLAB; behaviour changes only if a caller opts in. | Skim. |
| **`DEVIATION`** | **Intentionally different behaviour.** | **Yes.** |
| **`DEFECT`** | **MATLAB behaviour we believe is wrong.** An option reproduces it. | **Yes.** |

**Every entry names its test**: `UKP-07` is pinned by `tests/test_port.py::test_UKP_07`.
`tests/test_porting_doc.py` fails the build if an ID loses its test, or if a `DEVIATION` or `DEFECT`
is missing from the register below.

---

## Register: everything that does not behave like MATLAB

| ID | Status | What changes | Why | Decides |
|---|---|---|---|---|
| [`UKP-08`](#ukp-08) | DEVIATION | Seeded RNG, one stream per block of chains; the result depends on the seed and not on the number of CPUs | MATLAB's `parfor` workers draw from streams that depend on the pool, so no run is reproducible | Ronnakrit |
| [`UKP-02`](#ukp-02) | DEVIATION | Non-finite UK'37, non-positive `pstd` and 2-D input raise `ValueError` | MATLAB runs on and returns NaN or nonsense | Ronnakrit |
| [`UKP-12`](#ukp-12) | EXTENSION | `ens="spread"` takes 1000 ensemble members from all 500 chains | MATLAB's `ens` comes from **four** of the 500 chains (see the entry); `"matlab"` stays the default | Jess |
| [`UKP-09`](#ukp-09) | EXTENSION | `n_jobs` (default every CPU) and `backend` | MATLAB's `parfor` uses the pool it is given | — |
| [`UKF-03`](#ukf-03) | EXTENSION | `clip=False` keeps unclipped forward draws | MATLAB always clips, which stays the default | — |
| [`CS-03`](#cs-03) | DEVIATION | Mismatched `lat`/`lon` lengths raise `ValueError` | MATLAB fails later with an indexing error | — |

Nothing here changes the posterior the sampler targets, the proposal, the acceptance rule, or the
percentile convention. Every number `UK_predict.m` puts in `output` is computed the same way.

---

## The spline — `augknt`, `spmak`, `fnxtr`, `fnval`

These four Curve Fitting Toolbox calls appear in both `UK_predict.m` (lines 93–94, 109–112) and
`UK_forward.m` (lines 31–42).

### SPL-01
`EXACT` · `kn = augknt(knots, 3)` → `augknt(knots, 3)`. End knots repeated to multiplicity exactly
3; `[-0.4 23.6 29.6]` becomes `[-0.4 -0.4 -0.4 23.6 29.6 29.6 29.6]`, so the spline has four
coefficients, matching `bdraws`' four columns.

### SPL-02
`EQUIV` · `spmak(kn, b)` + `fnval` inside `[-0.4, 29.6]` → the quadratic B-spline on those knots.
Agrees with `scipy.interpolate.BSpline` to 1e-14. At the right end the value is the limit from the
left, as `fnval` returns, not the zero a right-continuous evaluator would give.

### SPL-03
`EQUIV` · `fnxtr(bs)` with its default order 2 → beyond the knots, the straight line matching the
spline's value and first derivative at the end. Without it, MATLAB notes, the B-spline is NaN
outside 0–30 °C.

### SPL-04
`EQUIV` · Evaluating 500 splines at once. MATLAB builds one spline object per chain. Here the spline
is written in the piecewise-polynomial form `fnxtr` itself produces (one polynomial per knot
interval, plus the two linear extensions), the coefficients for each chain are gathered with one
array operation, and the polynomial is evaluated by Horner's rule. Pinned against one-at-a-time
evaluation to 1e-13.

### SPL-05
`EQUIV` · Independent cross-check: `brews/baysplinepy` re-implemented `fnxtr` separately (via
`scipy.interpolate.PPoly`). The two agree to 2e-15 from −30 to 60 °C. Under MATLAB proper,
`test_vs_original.py::test_spline_exact` checks against the toolbox itself.

---

## `UK_predict.m`

### UKP-01
`EXACT` · Lines 58–67: `load('bayes_posterior_v2.mat')`, then `bdraws(1:2:end,:)`,
`tau2(1:2:end)` → `bdraws[::2]`, `tau2[::2]`. 500 of the 1000 stored draws; one Metropolis-Hastings
chain per draw. The optional third argument `bayes` is the `bayes=` keyword. The `.mat` file ships
inside the package and is tested byte-identical to the repository's (`STO-01`).

### UKP-02
`DEVIATION` · Line 70: `uk = uk(:)`. A row or column is accepted, as in MATLAB. A matrix is rejected
rather than flattened (MATLAB is column-major, NumPy row-major — `GEN-01`). NaN or infinite UK'37
raises: in MATLAB a NaN makes `median(prior_mean)` NaN, so the jump width is NaN and every proposal
is NaN for every point, not just the bad one. `pstd <= 0` raises.

### UKP-03
`EXACT` · Lines 79–86: `prior_mean = (uk - .039) ./ .034` (Prahl et al., 1988);
`prior_var = pstd^2`; chains start at the prior mean.

### UKP-04
`EXACT` · Lines 74–77: `N = 500` iterations, `burnin = 250`. The retained sample is iterations
251–500 of each chain.

### UKP-05
`EXACT` · Lines 98–106: the jump width from `median(prior_mean)` — 3.5 below 20 °C, 3.7 from 20 to
23.7 °C inclusive, `0.8092 * pmm - 15.1405` above. Tested on all three branches and both edges.

### UKP-06
`EXACT` · Lines 109–152, the chain. Start at the prior mean; propose `normrnd(s_now, JW)`; compute
`normpdf(uk, fnval(bs, s), sqrt(tau)) .* normpdf(s, prior_mean, pstd)` for proposal and current
state; accept where `rand <= min(1, ratio)`. The ratio is of densities, not log-densities, exactly
as MATLAB has it, so the two underflow in the same places. The Python moves the `jj` loop inside the
arrays — row `p` of each `(n_chains, N)` array is MATLAB's chain `jj` — and the test replays the
stream by hand to confirm the step.

### UKP-07
`EXACT` · Line 136: `min(1, MH_rat)`. MATLAB's `min` ignores NaN, so `min(1, 0/0)` is 1 and the
proposal is accepted. `np.minimum` propagates the NaN and would reject; `np.fmin` is MATLAB's `min`
and is what the port uses.

### UKP-08
`DEVIATION` · Randomness. MATLAB uses the global stream inside `parfor`, so results depend on the
pool and differ every run. The port cuts the 500 chains into 50 fixed blocks of 10, gives each a
child of one `numpy.random.SeedSequence(seed)`, and runs the blocks on as many CPUs as asked. **The
block layout does not depend on the CPU count**, so `seed=0` on one core and on sixty-four returns
bit-identical output — tested across 1, 2, all cores and the thread backend.

### UKP-09
`EXTENSION` · `parfor jj = 1:N_p` → `n_jobs`, default `-1`: every CPU this process may use (the
affinity mask, not the machine's count, so it behaves inside containers and batch allocations).
Processes, not threads: the inner loop is many small array operations, and threads measured three
times slower than one core because of the GIL. A script on macOS or Windows needs the usual
`if __name__ == "__main__":` guard; without one the port falls back to one core with a warning.

### UKP-10
`EQUIV` · Lines 157–161 and the `ChainConvergence` subfunction: Gelman's R-hat per point, the median
across points. Same arithmetic, vectorised over points; each worker returns its chains' means and
variances so the pooled sample is not swept twice. Tested against a literal transcription of
`ChainConvergence.m`, including its `kron`, to 1e-12.

*Note, no change:* each "chain" runs under a different posterior draw of the spline, so the chains
have different stationary distributions and R-hat includes calibration spread. Values slightly
above 1 (≈1.02 on the demo) are expected and are not a convergence failure.

### UKP-11
`EXACT` · Line 163: `MH_c = reshape(MH_samps_t, N_Ts, N_p*(N-burnin))`. Column-major, so pooled
column `c` is chain `c mod 500` at retained iteration `c div 500`. Internally the port keeps each
point's samples chain-major (the percentiles don't care about order) and reproduces MATLAB's order
wherever it matters — `ens` and `save_samples=True`. Tested by index.

### UKP-12
`EXTENSION` · Line 168: `output.ens = MH_c(:, 1:125:end)`. Because of `UKP-11`'s column order,
every 125th column is chains **1, 126, 251 and 376** at all 250 retained iterations — 1000 members
from four of the 500 spline draws. The comment says "save subsample of ensemble" and the README calls
it "the full ensemble", which suggests the intent was a spread across chains. `output.SST` is
unaffected: it is computed from all 125,000 samples.

`ens="matlab"` (the default) reproduces this exactly. `ens="spread"` takes iterations 1 and 126 of
every chain: still 1000 members, now covering all 500 draws. Measured on the demo record: MATLAB's
`ens` gives a 95% interval of 9.2 °C where the full sample gives 9.7 °C and `"spread"` 9.6 °C, and
the median of MATLAB's `ens` for one point moves 0.25 °C between seeds against 0.10 °C for
`"spread"`. Whether the default should change is Jess's call.

### UKP-13
`EXACT` · Lines 170–172: `sort(MH_c, 2)` and index `round([.025 .50 .975] .* 125000)` — order
statistics 3125, 62500 and 121875. Not `prctile`, and not interpolated. The port finds the same
three order statistics with `numpy.partition` (linear time, rows split over threads) instead of a
full sort; the test checks them against `np.sort` exactly.

### UKP-14
`EXACT` · Line 166: `accepts = nansum(accepts_t(:)) / (N_Ts*N_p*(N-burnin))`.

*Not ported:* lines 174–193, the two sanity-check figures. MATLAB draws them on every call; the
comparison notebook draws the same two, from the result.

---

## `UK_forward.m`

### UKF-01
`EXACT` · Lines 18–26: load the posterior; the optional `bayes` argument is `bayes=`. All 1000 draws,
not thinned.

### UKF-02
`EXACT` · Lines 31–49: one vector-valued spline over all draws, `fnval` at the SSTs (1000 × N), then
`normrnd(mean, sqrt(tau2))` and transpose to N × 1000. The test replays the draw.

### UKF-03
`EXTENSION` · Lines 51–52: clip to `[0, 1]`, the bounds of the index. `clip=True` is the default;
the share of clipped values is recorded either way in `metadata["clipped_fraction"]`.

### UKF-04
`EQUIV` · Round trip: forward-modelling three temperatures and predicting back recovers them to
within 1.5 °C. Not a MATLAB line; it pins the two functions as inverses of one another.

---

## `checkSeasonality.m`

### CS-01
`EXACT` · Lines 23–49. The three polygons and their months, in MATLAB's order — Mediterranean,
North Atlantic, North Pacific — so later regions override earlier ones. The North Pacific test
uses 0–360° longitude, so the polygon may cross the dateline. Returns a list of month arrays, the
Python form of MATLAB's cell array.

### CS-02
`EQUIV` · `inpolygon` counts points on an edge or vertex as inside. The port's ray-cast adds an
explicit on-edge test so boundary points (e.g. 48°N, 30°W on the North Atlantic polygon's southern
edge) match MATLAB.

### CS-03
`DEVIATION` · Mismatched `lat` and `lon` lengths raise `ValueError`.

---

## General

### STO-01
`EXACT` · The packaged `bayes_posterior_v2.mat` and `DemoUKData.csv` are byte-identical to the
repository root's.

### STO-02
`EXACT` · Another posterior file can be passed, as MATLAB's optional argument; it must carry
`bdraws`, `tau2` and `knots`.

### GEN-01
`EQUIV` · `x(:)`: a row or column becomes 1-D; a matrix is rejected.

### GEN-02
`EXACT` · MATLAB `round` is half away from zero; NumPy's is half to even.

### GEN-03
`EQUIV` · `normpdf` is the Statistics Toolbox formula, `exp(-z²/2) / (σ√(2π))`.

---

## Compared with `brews/baysplinepy`

S. B. Malevich's [`baysplinepy`](https://github.com/brews/baysplinepy) 0.0.2 is the existing Python
port. `BAYSPLINEpy/notebooks/compare_implementations.ipynb` runs it beside this one and the MATLAB
reference. What differs, and what each difference costs:

| | `brews/baysplinepy` 0.0.2 | this port |
|---|---|---|
| **Posterior file** | `bayes_posterior.mat`: an earlier fit, 1500 draws thinned 1:3 | `bayes_posterior_v2.mat`, what MATLAB loads, thinned 1:2. Both give 500 chains; the medians they produce agree within noise |
| **Prior inside the sampler** | `sample_now = initial_sst` aliases `prior_mean`, so every accepted move also **moves the prior mean**, and each chain starts where the previous one ended | MATLAB's fixed prior |
| **Effect, demo record** | median **+0.44 °C** warm, 95% interval **34% wider** (upper bound +3.3 °C), R-hat 1.16 | matches the reference within noise; R-hat 1.02 |
| **Effect, UK'37 0.4–0.6** | within noise — the likelihood dominates there | — |
| **Percentiles** | `.percentile()` defaults to 5/50/95 with `interpolation='nearest'`, which **raises on NumPy ≥ 2.0** | `output.SST`: 2.5/50/97.5 by order statistic, as MATLAB |
| **Forward model** | no clip to [0, 1]; 1500 draws | clipped, 1000 draws, as MATLAB |
| **Speed, demo (97 points)** | ~13 s, one core | ~1 s on 4 cores, ~3.5 s on one |
| **SciPy** | imports `scipy.interpolate.fitpack.splev`, removed in SciPy 2.0 | public API only |
| **checkSeasonality** | not ported | ported |

The aliasing is one line: `sample_now = initial_sst` should be `initial_sst.copy()`. With that
change alone his port agrees with this one and with MATLAB, which is the strongest evidence that it
is the cause.
