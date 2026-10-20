"""The B-spline, as MATLAB's Curve Fitting Toolbox builds it (PORTING.md SPL-01..04).

``UK_predict.m`` and ``UK_forward.m`` both do::

    kn = augknt(knots, 3);          % order 3 = quadratic
    bs = fnxtr(spmak(kn, coefs));   % linear beyond the knots
    mean = fnval(bs, sst);

A spline with fixed knots is linear in its coefficients, so the mean can be
written ``mean = W(sst) @ coefs`` where the design row ``W`` depends on the
temperature alone. That is what lets the sampler evaluate all 500 posterior
splines in one array operation instead of building 500 spline objects.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

#: Spline order in MATLAB's sense (degree + 1). UK_predict.m:93, UK_forward.m:31.
ORDER = 3


def augknt(knots, order: int = ORDER) -> np.ndarray:
    """MATLAB ``augknt(knots, k)`` (PORTING.md SPL-01).

    Returns the knots, sorted with duplicates removed, with the first and last
    repeated to multiplicity exactly `order`. Interior knots keep multiplicity
    one, which is all ``augknt`` does with a two-argument call.

    Args:
        knots: Breakpoints (``bayes_posterior_v2.mat`` holds ``[-0.4 23.6 29.6]``).
        order: Spline order, degree + 1.

    Returns:
        The augmented knot sequence.
    """
    k = np.unique(np.asarray(knots, dtype=float).ravel())
    return np.concatenate([np.repeat(k[0], order), k[1:-1], np.repeat(k[-1], order)])


class Spline:
    """``fnxtr(spmak(augknt(knots, order), ·))`` for any set of coefficients.

    Inside the basic interval ``[t_first, t_last]`` this is the B-spline. Outside
    it, ``fnxtr``'s default order 2 continues the spline by the straight line
    that matches its value and first derivative at the end (PORTING.md SPL-03).

    Args:
        knots: Breakpoints, before augmentation.
        order: Spline order, degree + 1.
    """

    def __init__(self, knots, order: int = ORDER) -> None:
        self.order = order
        self.degree = order - 1
        self.t = augknt(knots, order)
        self.n_coefs = len(self.t) - order
        self.lo, self.hi = float(self.t[0]), float(self.t[-1])
        eye = np.eye(self.n_coefs)
        basis = [BSpline(self.t, e, self.degree, extrapolate=False) for e in eye]
        # Value and slope of each basis function at the two ends. The right end
        # is evaluated as a limit from the left, which is what makes fnval's
        # value there the last coefficient rather than zero.
        self._val_lo = np.array([b(self.lo) for b in basis])
        self._val_hi = np.array([BSpline(self.t, e, self.degree)(self.hi) for e in eye])
        self._der_lo = np.array([BSpline(self.t, e, self.degree).derivative()(self.lo)
                                 for e in eye])
        self._der_hi = np.array([BSpline(self.t, e, self.degree).derivative()(self.hi)
                                 for e in eye])
        # The ppform fnxtr builds: one polynomial per piece, in powers of
        # (x - origin). Pieces are the two linear extensions plus one per knot
        # interval. _pp maps coefficients to the Taylor terms at each origin.
        breaks = np.unique(self.t)
        self.breaks = breaks
        self.origins = np.concatenate([[self.lo], breaks[:-1], [self.hi]])
        pp = np.zeros((len(self.origins), order, self.n_coefs))
        pp[0, 0], pp[0, 1] = self._val_lo, self._der_lo
        for p_, x0 in enumerate(breaks[:-1], start=1):
            for m in range(order):
                pp[p_, m] = [BSpline(self.t, e, self.degree).derivative(m)(x0) if m else
                             BSpline(self.t, e, self.degree)(x0) for e in eye]
                pp[p_, m] /= np.prod(np.arange(1, m + 1))
        pp[-1, 0], pp[-1, 1] = self._val_hi, self._der_hi
        self._pp = pp

    def basis(self, x):
        """The nonzero B-splines at each point (de Boor; Piegl & Tiller A2.2).

        Only ``order`` basis functions are nonzero at any point, so this returns
        those and where they start rather than a dense row. Points outside the
        basic interval are clamped to it; :meth:`__call__` replaces their values
        with the linear extension.

        Args:
            x: Temperatures, any shape.

        Returns:
            ``(first, vals)``: the index of the first nonzero coefficient, shape
            ``x.shape``, and the basis values, shape ``x.shape + (order,)``.
        """
        p, t = self.degree, self.t
        u = np.clip(np.asarray(x, dtype=float), self.lo, self.hi)
        # The knot interval [t[i], t[i+1]) holding u, with the right end folded
        # into the last interval so fnval's left limit is what comes back there.
        i = np.clip(np.searchsorted(t, u, side="right") - 1, p, self.n_coefs - 1)
        vals = [np.ones_like(u)]
        left, right = [None], [None]
        for j in range(1, p + 1):
            left.append(u - t[i + 1 - j])
            right.append(t[i + j] - u)
            saved = np.zeros_like(u)
            for r in range(j):
                temp = vals[r] / (right[r + 1] + left[j - r])
                vals[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            vals.append(saved)
        return i - p, np.stack(vals, axis=-1)

    def design(self, x) -> np.ndarray:
        """The design rows ``W`` such that ``fnval(bs, x) == W @ coefs``.

        Args:
            x: Temperatures, any shape.

        Returns:
            Array of shape ``x.shape + (n_coefs,)``.
        """
        x = np.asarray(x, dtype=float)
        first, vals = self.basis(x)
        out = np.zeros(x.shape + (self.n_coefs,))
        for r in range(self.order):
            np.put_along_axis(out, (first + r)[..., None], vals[..., r:r + 1], axis=-1)
        below, above = x < self.lo, x > self.hi
        out[below] = self._val_lo + (x[below] - self.lo)[:, None] * self._der_lo
        out[above] = self._val_hi + (x[above] - self.hi)[:, None] * self._der_hi
        out[np.isnan(x)] = np.nan
        return out

    def __call__(self, x, coefs) -> np.ndarray:
        """Evaluate one spline, or many at once.

        Args:
            x: Temperatures. With 2-D `coefs`, shape ``(n_splines, ...)`` gives
                each spline its own points; a 1-D `x` is shared by all.
            coefs: ``(n_coefs,)`` for one spline, ``(n_splines, n_coefs)`` for many.

        Returns:
            For one spline, ``x.shape``. For many with shared `x`,
            ``(n_splines, len(x))`` — ``fnval``'s layout for a vector-valued
            spline. For many with their own points, ``x.shape``.
        """
        coefs = np.asarray(coefs, dtype=float)
        x = np.asarray(x, dtype=float)
        if coefs.ndim == 1:
            return self.design(x) @ coefs
        if x.ndim <= 1:
            return coefs @ self.design(np.atleast_1d(x)).T
        # One spline per leading row. Evaluate the ppform: find each point's
        # piece, gather that row's polynomial with one flat take per power, and
        # Horner. Far cheaper than de Boor here, where this is the inner loop.
        n_rows, n_pieces = coefs.shape[0], len(self.origins)
        terms = np.einsum("pmj,rj->rpm", self._pp, coefs).reshape(n_rows * n_pieces, self.order)
        piece = np.searchsorted(self.breaks, x, side="right")
        rows = np.arange(n_rows).reshape((n_rows,) + (1,) * (x.ndim - 1)) * n_pieces
        flat = (rows + piece).ravel()
        dx = (x - self.origins[piece]).ravel()
        out = np.take(terms[:, -1], flat)
        for m in range(self.order - 2, -1, -1):
            out = out * dx + np.take(terms[:, m], flat)
        return out.reshape(x.shape)
