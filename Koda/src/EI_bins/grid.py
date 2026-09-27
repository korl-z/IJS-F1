"""Shell grid in the bare band variable s = cos kx + cos ky (2d, a0 = 1).

Both bare bands depend on k only through s, so E, u, v of the quasiparticles
are functions of s for any (d, m). Cells are equally spaced in s, hence in the
bare energies, and never move during the self consistent solve.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class Band:
    """2d nn tight binding: ea = gap/2 - 2 ta s, eb = -gap/2 - 2 tb s."""

    gap: float
    ta: float
    tb: float

    def bare(self, s):
        ea = 0.5 * self.gap - 2.0 * self.ta * s
        eb = -0.5 * self.gap - 2.0 * self.tb * s
        return ea, eb

    @property
    def slope(self):
        # upper bound of |dE/ds| for both quasiparticle branches
        return abs(self.ta - self.tb) + abs(self.ta + self.tb)


@dataclass(frozen=True)
class MF:
    """Same fields as eu.MFPars, so either can be passed."""

    v: float
    n: float = 1.0
    uh: float | None = None

    @property
    def h(self):
        return self.v if self.uh is None else float(self.uh)


@dataclass(frozen=True)
class Shells:
    se: np.ndarray  # cell edges in s, (ns + 1,)
    sc: np.ndarray  # cell centers, (ns,)
    w: np.ndarray  # BZ fraction per cell, sums to 1
    ss: np.ndarray  # sub node positions, (ns, nsub)
    ws: np.ndarray  # sub node weights, normalized per cell

    @property
    def ns(self):
        return self.sc.size

    def cell(self, s):
        """Cell index of arbitrary s values."""
        i = np.searchsorted(self.se, s, side="right") - 1
        return np.clip(i, 0, self.ns - 1)


def dos_cdf(x, nq=200):
    """Fraction of the BZ with cos kx + cos ky <= x.

    F(x) = 1 - (1/pi^2) int_0^pi arccos(clip(x - cos k)) dk, split at the
    kinks x - cos k = +-1 and integrated with Gauss-Legendre.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t, wq = leggauss(nq)
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        br = [0.0, np.pi]
        for c in (xi - 1.0, xi + 1.0):
            if -1.0 < c < 1.0:
                br.append(np.arccos(c))
        br = np.unique(br)
        tot = 0.0
        for a, b in zip(br[:-1], br[1:]):
            k = 0.5 * (b - a) * t + 0.5 * (a + b)
            f = np.arccos(np.clip(xi - np.cos(k), -1.0, 1.0))
            tot += 0.5 * (b - a) * np.dot(wq, f)
        out[i] = 1.0 - tot / np.pi**2
    return out


def make_shells(ns, nsub=4, nq=200):
    """Uniform cells on s in [-2, 2]; ns odd keeps s = 0 at a cell center."""
    ns = int(ns)
    if ns < 3 or ns % 2 == 0:
        raise ValueError("ns must be odd and at least 3")
    fe = np.linspace(-2.0, 2.0, ns * nsub + 1)
    cf = dos_cdf(fe, nq)
    cf[0], cf[-1] = 0.0, 1.0
    # exact s -> -s symmetry of the square lattice
    cf = 0.5 * (cf + 1.0 - cf[::-1])
    wf = np.diff(cf).reshape(ns, nsub)
    ss = 0.5 * (fe[:-1] + fe[1:]).reshape(ns, nsub)
    w = wf.sum(axis=1)
    if np.any(w <= 0.0):
        raise RuntimeError("empty cell, check ns")
    return Shells(fe[::nsub].copy(), ss.mean(axis=1), w / w.sum(), ss, wf / w[:, None])


def auto_ns(band, scale, r=4.0, nmin=21):
    """Odd ns such that a cell spans at most scale / r in quasiparticle energy.

    scale is the smallest physical energy to resolve, e.g. min(d0, Tc, w0).
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    ds = scale / (r * band.slope)
    ns = max(int(nmin), int(np.ceil(4.0 / ds)))
    return ns + 1 - ns % 2


def s_of_k(k):
    """Shell variable of momenta with shape (..., 2)."""
    k = np.asarray(k, dtype=float)
    return np.cos(k[..., 0]) + np.cos(k[..., 1])
