"""``brews/baysplinepy``'s names, over this package's sampler.

``predict_sst(uk, prior_std)`` and ``predict_uk(sst)`` take the arguments his
functions do, so a notebook ports by changing ``import bayspline`` to
``import baysplinepy``. The numbers follow the MATLAB reference, not his port —
see ``docs/BAYSPLINEpy/PORTING.md`` §"Compared with brews/baysplinepy".
"""
from __future__ import annotations

from .results import SSTPrediction, UKPrediction
from .uk_forward import uk_forward
from .uk_predict import uk_predict


def predict_sst(uk, prior_std: float, progressbar: bool = False, *, seed=None,
                n_jobs: int | None = -1, **kwargs) -> SSTPrediction:
    """SST from UK'37 (``brews/baysplinepy`` signature).

    Args:
        uk: (N,) UK'37 values.
        prior_std: Prior standard deviation, °C.
        progressbar: Accepted for compatibility and ignored — the run takes a
            second or two.
        seed: Seed for the sampler.
        n_jobs: Workers; ``-1`` is every CPU.
        **kwargs: Passed to :func:`~baysplinepy.uk_predict.uk_predict`.

    Returns:
        An :class:`~baysplinepy.results.SSTPrediction`.
    """
    del progressbar
    return uk_predict(uk, prior_std, seed=seed, n_jobs=n_jobs, **kwargs)


def predict_uk(sst, *, seed=None, **kwargs) -> UKPrediction:
    """UK'37 from SST (``brews/baysplinepy`` signature).

    Args:
        sst: (N,) SSTs, °C.
        seed: Seed for the draw.
        **kwargs: Passed to :func:`~baysplinepy.uk_forward.uk_forward`.

    Returns:
        A :class:`~baysplinepy.results.UKPrediction`.
    """
    return uk_forward(sst, seed=seed, **kwargs)
