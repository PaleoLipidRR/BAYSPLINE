# BAYSPLINEpy

A Python recode of the MATLAB BAYSPLINE package in the repository above: a Bayesian calibration for
the alkenone (U<sup>K'</sup><sub>37</sub>) paleothermometer (Tierney & Tingley, 2018,
[doi:10.1002/2017PA003201](https://doi.org/10.1002/2017PA003201)).

It runs **on every CPU at once**, as MATLAB's `parfor` does, and its answer depends on the seed
alone, not on how many CPUs ran it.

**Prototype.** It lives inside the MATLAB repository, beside the `.m` files it is checked against,
the same way BAYSPARpy does in `PaleoLipidRR/BAYSPAR`. Nothing is published to PyPI.

```bash
cd BAYSPLINEpy
pip install -e ".[dev]"          # or: PYTHONPATH=src python -m pytest
python -m pytest                 # 51 tests
```

The import package is `baysplinepy`, not `bayspline`: `brews/baysplinepy` already imports as
`bayspline`, and the comparison notebook installs both.

## Using it

```python
import baysplinepy as bsp

# DemoScript.m
demo = bsp.load_demo_data()                       # DemoUKData.csv
out = bsp.uk_predict(demo["uk37"], 5, seed=0)     # UK_predict.m; every CPU by default
out.sst            # (N, 3): 2.5th, 50th, 97.5th percentiles, °C  (output.SST)
out.ensemble       # (N, 1000)                                     (output.ens)
out.prior_mean, out.jump_dist, out.rhat, out.accepts
out.to_dataframe().to_csv("DemoCalibratedData.csv")   # needs pandas

bsp.uk_predict(uk, 5, n_jobs=1)          # one core
bsp.uk_predict(uk, 5, n_jobs=-2)         # all but one
bsp.uk_predict(uk, 5, ens="spread")      # ensemble from all 500 chains (PORTING.md UKP-12)

# UK_forward.m: 1000 UK'37 draws per SST, clipped to [0, 1]
fwd = bsp.uk_forward([10.0, 20.0, 28.0], seed=0)
fwd.uk.shape                             # (3, 1000)

# checkSeasonality.m
bsp.check_seasonality(lat=[60, 40], lon=[-20, 15])   # [array([8, 9, 10]), array([1, 2, 3, 4, 5, 11, 12])]
```

`predict_sst(uk, prior_std)` and `predict_uk(sst)` are the same functions under
`brews/baysplinepy`'s names, so his notebooks port by changing the import.

**Scripts on macOS and Windows** start worker processes by re-importing the script, so put the
call under `if __name__ == "__main__":`. Without it the port falls back to one core with a warning.
Notebooks need nothing.

## Speed

The Gulf of Aden demo record (97 points; 500 chains × 500 iterations, as MATLAB does it):

| | wall clock |
|---|---|
| `brews/baysplinepy` 0.0.2 | ~13 s, one core |
| BAYSPLINEpy, `n_jobs=1` | ~3.5 s |
| BAYSPLINEpy, 4 CPUs | ~1.1 s |

The README of the MATLAB package quotes 20–90 s. The output is bit-identical for any `n_jobs`
(`tests/test_port.py::test_UKP_08`).

## Compared with MATLAB and `brews/baysplinepy`

`notebooks/compare_implementations.ipynb` runs all three on the demo record and two other cases.
In short:

- **Against MATLAB:** prior means, jump width, the spline and `checkSeasonality` agree exactly; the
  SST percentiles, acceptance rate and R-hat agree within Monte-Carlo noise
  (`tests/test_vs_original.py`, against `UK_predict.m` executing).
- **`brews/baysplinepy` runs warm and wide at high UK'37.** `sample_now = initial_sst` makes the
  sampler's state and the prior mean one array, so every accepted step moves the prior. On the demo
  record its median comes out ~0.4 °C warm and its 97.5th percentile ~3 °C warm (R-hat 1.16 against
  MATLAB's ~1.02). Changing that line to `initial_sst.copy()` brings it into agreement. At UK'37
  0.4–0.6, the range his tests use, the effect is within noise.
- **MATLAB's `output.ens` comes from 4 of the 500 chains**, a side effect of a column-major
  `reshape`. Reproduced by default; `ens="spread"` is the alternative (PORTING.md UKP-12).

```bash
uv sync                                  # .venv with the package, dev tools and brews/baysplinepy
uv run jupyter lab notebooks/compare_implementations.ipynb
```

or `pip install -e ".[dev,compare]"`. The notebook runs without brews/baysplinepy and says so.
Its source is `notebooks/build_notebook.py`.

## Reviewing it

`../docs/BAYSPLINEpy/PORTING.md` walks every line of the three MATLAB functions under a stable ID and
says what this code does instead. Six entries change behaviour or add an option and sit in a
register at the top. `tests/test_port.py` has one test per ID, and `tests/test_porting_doc.py` fails
the build if an entry loses its test or escapes the register.

The MATLAB reference, `../docs/BAYSPLINEpy/audit/original_matlab.mat`, was made by
`audit/verify_original.m` under GNU Octave 8.4. Octave has no Curve Fitting Toolbox, so the script
adds `audit/octave_shims/` (`augknt`, `spmak`, `fnxtr`, `fnval`) only when those functions are
missing. Every other line that ran is the original `.m` source. Re-running under MATLAB removes the
caveat:

```bash
matlab -batch "addpath('docs/BAYSPLINEpy/audit'); verify_original"
```
