"""Small translations shared by several functions (PORTING.md GEN-01..03)."""
from __future__ import annotations

import numpy as np

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def as_column(x, name: str = "input") -> np.ndarray:
    """MATLAB ``x = x(:)`` — flatten to a 1-D float array (GEN-01).

    MATLAB flattens column-major and NumPy row-major, so anything with more than
    one non-trivial dimension is rejected rather than flattened differently.

    Args:
        x: Scalar or array-like.
        name: Name to use in the error message.

    Returns:
        A 1-D float64 array.

    Raises:
        ValueError: if `x` has more than one non-trivial dimension.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1 and sum(d > 1 for d in arr.shape) > 1:
        raise ValueError(
            f"{name} has shape {arr.shape}; MATLAB's (:) flattens column-major and "
            "NumPy row-major, so pass a 1-D sequence to make the order explicit"
        )
    return arr.reshape(-1)


def matlab_round(x) -> np.ndarray:
    """MATLAB ``round``: halves away from zero (GEN-02).

    :func:`numpy.round` rounds halves to even, which picks a different index
    wherever a product lands on .5.

    Args:
        x: Array-like.

    Returns:
        Rounded array, float.
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def normpdf(x, mu, sigma) -> np.ndarray:
    """MATLAB ``normpdf(x, mu, sigma)`` (GEN-03).

    ``exp(-0.5 z^2) / (sqrt(2 pi) sigma)``, the expression the Statistics
    Toolbox evaluates; :func:`scipy.stats.norm.pdf` gives the same numbers
    about forty times slower.
    """
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (_SQRT_2PI * sigma)
