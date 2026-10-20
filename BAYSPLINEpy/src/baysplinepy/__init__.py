"""BAYSPLINE — a Bayesian calibration for the alkenone paleothermometer.

A Python recode of the MATLAB package of Tierney & Tingley (2018). The
translation is documented line by line in ``docs/BAYSPLINEpy/PORTING.md``; each
deliberate difference from MATLAB carries an ID quoted in the relevant docstring.

Cite the original methodology:

    Tierney, J. E., & Tingley, M. P. (2018). BAYSPLINE: A new calibration for the
    alkenone paleothermometer. Paleoceanography and Paleoclimatology, 33,
    281-301. https://doi.org/10.1002/2017PA003201
"""
from importlib.metadata import PackageNotFoundError as _NotFound, version as _version

from .convergence import chain_convergence, effective_size
from .demo import load_demo_data
from .parallel import available_cpus
from .posterior import Posterior, load_posterior
from .predict import predict_sst, predict_uk
from .results import SSTPrediction, UKPrediction
from .seasonality import check_seasonality, inpolygon
from .spline import Spline, augknt
from .uk_forward import uk_forward
from .uk_predict import jump_width, uk_predict

try:
    __version__ = _version("bayspline")
except _NotFound:      # running from a source checkout that was never installed
    __version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    # MATLAB-faithful layer
    "uk_predict", "uk_forward", "check_seasonality",
    # brews/baysplinepy's names
    "predict_sst", "predict_uk",
    # results
    "SSTPrediction", "UKPrediction",
    # data
    "Posterior", "load_posterior", "load_demo_data",
    # pieces worth reusing
    "Spline", "augknt", "chain_convergence", "effective_size", "inpolygon",
    "jump_width", "available_cpus",
]
