import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from scipy.optimize import brentq, root
from scipy.special import expit

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import matplotlib.cm as cm
import colormaps as cmaps
import matplotlib.colors as colors
from labellines import labelLines
from matplotlib.cm import brg, ScalarMappable
from matplotlib.colors import (
    LogNorm,
    TwoSlopeNorm,
    LinearSegmentedColormap,
    Normalize,
    ListedColormap,
)
from matplotlib.lines import Line2D

from dataclasses import dataclass
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
from jax import random
from jax import vmap

W = 3.47412


def newfig(h=0.6):
    fig, ax = plt.subplots(figsize=(W, h * W))
    ax.tick_params(which="both", direction="in")
    return fig, ax


# fermi dirac (E, mu, T)
def fd(e, mu, t):
    te = max(t, 1e-5)
    return expit((mu - e) / te)




Arr = NDArray[np.float64]

__version__ = "0.2.0"

__all__ = [
    "Bands",
    "Pars",
    "Obs",
    "Sol",
    "NSol",
    "bands",
    "from_fn",
    "toy",
    "sq",
    "fd",
    "data",
    "obs",
    "res",
    "solve",
    "nres",
    "normal",
    "pair",
    "crit",
    "mu_root",
    "loop",
    "follow",
]


@dataclass(frozen=True)
class Bands:
    """Bare bands and quadrature weights."""
    ea: Arr
    eb: Arr
    w: Arr

    def __post_init__(self) -> None:
        ea = np.asarray(self.ea, dtype=float)
        eb = np.asarray(self.eb, dtype=float)
        w = np.asarray(self.w, dtype=float)

        if ea.shape != eb.shape or ea.shape != w.shape:
            raise ValueError("ea, eb, and w must have the same shape")
        if ea.size == 0 or not np.all(np.isfinite(ea + eb + w)):
            raise ValueError("band data must be finite and nonempty")
        if np.any(w < 0.0) or np.sum(w) <= 0.0:
            raise ValueError("weights must be nonnegative with positive sum")

        object.__setattr__(self, "ea", ea.ravel())
        object.__setattr__(self, "eb", eb.ravel())
        object.__setattr__(self, "w", w.ravel())


@dataclass(frozen=True)
class Pars:
    """SCF parameters in one common energy unit."""

    v: float
    t: float
    n: float = 1.0
    uh: float | None = None

    @property
    def h(self) -> float:
        return self.v if self.uh is None else self.uh


@dataclass(frozen=True)
class Obs:
    """Mean-field output for one trial state."""

    d: float
    m: float
    n: float
    na: float
    nb: float


@dataclass(frozen=True)
class Sol:
    """Converged state."""

    d: float
    m: float
    mu: float
    na: float
    nb: float
    err: float
    nit: int

    @property
    def y(self) -> Arr:
        return np.array([self.d, self.m, self.mu])


@dataclass(frozen=True)
class NSol:
    """Converged normal state."""

    m: float
    mu: float
    na: float
    nb: float
    err: float
    nit: int

    @property
    def z(self) -> Arr:
        return np.array([self.m, self.mu])


def bands(ea: ArrayLike, eb: ArrayLike, w: ArrayLike | None = None) -> Bands:
    """Build band data. Uniform weights sum to one by default."""

    aa = np.asarray(ea, dtype=float)
    bb = np.asarray(eb, dtype=float)
    if w is None:
        ww = np.full(aa.shape, 1.0 / aa.size)
    else:
        ww = np.asarray(w, dtype=float)
    return Bands(aa, bb, ww)


def from_fn(q: ArrayLike, fn: Callable[[Arr], tuple[Arr, Arr]],
            w: ArrayLike | None = None) -> Bands:
    """Build bands from any grid and dispersion callback."""

    qq = np.asarray(q, dtype=float)
    ea, eb = fn(qq)
    return bands(ea, eb, w)


def toy(nx: int, wd: float, g: float, r: float = 1.0,
        g0: float | None = None) -> Bands:
    """Finite energy-window model used in the earlier examples."""

    dx = wd / nx
    x = (np.arange(nx) + 0.5) * dx
    den = 1.0 / wd if g0 is None else g0
    w = np.full(nx, den * dx)
    ea = 0.5 * g + x
    eb = -0.5 * g - r * x
    return bands(ea, eb, w)


def sq(nk: int, d: float, ta: float, tb: float,
       a0: float = 1.0) -> Bands:
    """Nearest-neighbor square-lattice model."""

    k = np.linspace(-np.pi / a0, np.pi / a0, nk, endpoint=False)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    c = np.cos(kx * a0) + np.cos(ky * a0)
    ea = 0.5 * d - 2.0 * ta * c
    eb = -0.5 * d - 2.0 * tb * c
    return bands(ea, eb)


def fd(e: ArrayLike, mu: float, t: float) -> Arr:
    """Fermi function with a small zero-temperature regularizer."""

    te = max(float(t), 1.0e-10)
    return expit((mu - np.asarray(e, dtype=float)) / te)


def data(bd: Bands, p: Pars, y: ArrayLike) -> dict[str, Arr | float]:
    """Diagonalize the mean-field Hamiltonian for one trial state."""

    d, m, mu = np.asarray(y, dtype=float)
    na = 0.5 * (p.n + m)
    nb = 0.5 * (p.n - m)

    ea = bd.ea + p.h * nb
    eb = bd.eb + p.h * na
    et = 0.5 * (ea + eb)
    xi = 0.5 * (ea - eb)
    ek = np.sqrt(xi * xi + d * d)
    ep = et + ek
    em = et - ek
    fp = fd(ep, mu, p.t)
    fm = fd(em, mu, p.t)

    z = np.divide(xi, ek, out=np.zeros_like(xi), where=ek > 1.0e-14)
    u2 = 0.5 * (1.0 + z)
    v2 = 1.0 - u2

    q = np.empty_like(ek)
    ix = ek > 1.0e-12
    q[ix] = (fm[ix] - fp[ix]) / (2.0 * ek[ix])
    f0 = fd(et[~ix], mu, p.t)
    q[~ix] = f0 * (1.0 - f0) / max(p.t, 1.0e-10)

    return {
        "ea": ea,
        "eb": eb,
        "ep": ep,
        "em": em,
        "fp": fp,
        "fm": fm,
        "u2": u2,
        "v2": v2,
        "q": q,
    }


def obs(bd: Bands, p: Pars, y: ArrayLike) -> Obs:
    """Compute the right-hand sides of the three SCF equations."""

    d = float(np.asarray(y, dtype=float)[0])
    z = data(bd, p, y)
    fp = np.asarray(z["fp"])
    fm = np.asarray(z["fm"])
    u2 = np.asarray(z["u2"])
    v2 = np.asarray(z["v2"])
    q = np.asarray(z["q"])

    na = float(np.sum(bd.w * (u2 * fp + v2 * fm)))
    nb = float(np.sum(bd.w * (v2 * fp + u2 * fm)))
    dd = float(p.v * d * np.sum(bd.w * q))
    return Obs(dd, na - nb, na + nb, na, nb)


def res(y: ArrayLike, bd: Bands, p: Pars) -> Arr:
    """Residuals for delta, orbital polarization, and filling."""

    d, m, _ = np.asarray(y, dtype=float)
    o = obs(bd, p, y)
    es = max(np.ptp(bd.ea), np.ptp(bd.eb), abs(p.v), abs(p.h), 1.0e-12)
    return np.array([(d - o.d) / es, m - o.m, o.n - p.n])


def solve(bd: Bands, p: Pars, y0: ArrayLike = (0.1, 0.0, 0.0),
          tol: float = 1.0e-10, nmax: int = 1000) -> Sol:
    """Solve the three coupled equations with a multidimensional root finder."""

    z = root(res, np.asarray(y0, dtype=float), args=(bd, p),
             method="hybr", options={"xtol": tol, "maxfev": nmax})
    er = float(np.max(np.abs(res(z.x, bd, p))))
    if not z.success or er > 10.0 * tol:
        raise RuntimeError(f"SCF failed: {z.message}; residual={er:.3e}")

    d, m, mu = z.x
    if d < 0.0:
        d = -d
    if d < 1.0e-9:
        d = 0.0
    o = obs(bd, p, (d, m, mu))
    return Sol(float(d), float(m), float(mu), o.na, o.nb, er, int(z.nfev))


def nres(z: ArrayLike, bd: Bands, p: Pars) -> Arr:
    """Normal-state residuals for polarization and filling."""

    m, mu = np.asarray(z, dtype=float)
    o = obs(bd, p, (0.0, m, mu))
    return np.array([m - o.m, o.n - p.n])


def normal(bd: Bands, p: Pars, z0: ArrayLike = (0.0, 0.0),
           tol: float = 1.0e-10, nmax: int = 1000) -> NSol:
    """Solve the normal state at fixed temperature and filling."""

    z = root(nres, np.asarray(z0, dtype=float), args=(bd, p),
             method="hybr", options={"xtol": tol, "maxfev": nmax})
    er = float(np.max(np.abs(nres(z.x, bd, p))))
    if not z.success or er > 10.0 * tol:
        raise RuntimeError(f"Normal solve failed: {z.message}; residual={er:.3e}")

    m, mu = z.x
    o = obs(bd, p, (0.0, m, mu))
    return NSol(float(m), float(mu), o.na, o.nb, er, int(z.nfev))


def pair(bd: Bands, p: Pars, z0: ArrayLike = (0.0, 0.0)) -> tuple[float, NSol]:
    """Return the normal-state pairing eigenvalue and state."""

    s = normal(bd, p, z0=z0)
    q = np.asarray(data(bd, p, (0.0, s.m, s.mu))["q"])
    la = float(p.v * np.sum(bd.w * q))
    return la, s


def crit(bd: Bands, v: float, n: float = 1.0, uh: float | None = None,
         tlo: float = 1.0e-4, thi: float = 2.0, nt: int = 80,
         z0: ArrayLike = (0.0, 0.0), tol: float = 1.0e-8,
         nmax: int = 100) -> float:
    """Find the highest continuous transition temperature."""

    if not 0.0 < tlo < thi:
        raise ValueError("Require 0 < tlo < thi")
    if nt < 2:
        raise ValueError("nt must be at least two")

    ta = np.geomspace(thi, tlo, nt)
    za = []
    ga = []
    z = np.asarray(z0, dtype=float)

    for t in ta:
        p = Pars(v=v, t=float(t), n=n, uh=uh)
        la, s = pair(bd, p, z0=z)
        z = s.z
        za.append(z.copy())
        ga.append(la - 1.0)

    ga = np.asarray(ga)
    if ga[0] >= 0.0:
        raise ValueError("thi is below the transition")

    ix = np.flatnonzero((ga[:-1] <= 0.0) & (ga[1:] >= 0.0))
    if ix.size == 0:
        return float("nan")

    j = int(ix[0])
    th, tl = float(ta[j]), float(ta[j + 1])
    gh, gl = float(ga[j]), float(ga[j + 1])
    zh, zl = za[j], za[j + 1]

    for _ in range(nmax):
        tm = 0.5 * (th + tl)
        zm = 0.5 * (zh + zl)
        p = Pars(v=v, t=tm, n=n, uh=uh)
        la, s = pair(bd, p, z0=zm)
        gm = la - 1.0

        if gm >= 0.0:
            tl, gl, zl = tm, gm, s.z
        else:
            th, gh, zh = tm, gm, s.z

        if th - tl <= tol * max(1.0, tm):
            return 0.5 * (th + tl)

    raise RuntimeError("Critical-temperature bisection did not converge")


def mu_root(bd: Bands, p: Pars, d: float, m: float) -> float:
    """Find the chemical potential at fixed delta and polarization."""

    z = data(bd, p, (d, m, 0.0))
    em = np.asarray(z["em"])
    ep = np.asarray(z["ep"])
    es = max(np.ptp(bd.ea), np.ptp(bd.eb), abs(p.v), abs(p.h), p.t, 1.0)
    lo = float(min(em.min(), ep.min()) - 20.0 * es)
    hi = float(max(em.max(), ep.max()) + 20.0 * es)

    def fn(mu: float) -> float:
        return float(np.sum(bd.w * (fd(ep, mu, p.t) + fd(em, mu, p.t))) - p.n)

    return float(brentq(fn, lo, hi))


def loop(bd: Bands, p: Pars, y0: ArrayLike = (0.1, 0.0, 0.0),
         mix: tuple[float, float, float] = (0.3, 0.3, 0.3),
         tol: float = 1.0e-9, nmax: int = 5000) -> Sol:
    """Run an explicit damped SCF loop."""

    y = np.asarray(y0, dtype=float).copy()
    a = np.asarray(mix, dtype=float)
    if np.any(a <= 0.0) or np.any(a > 1.0):
        raise ValueError("mix entries must lie in (0, 1]")

    for it in range(1, nmax + 1):
        d, m, _ = y
        mu = mu_root(bd, p, d, m)
        o = obs(bd, p, (d, m, mu))
        yr = np.array([o.d, o.m, mu])
        yn = (1.0 - a) * y + a * yr
        er = float(np.max(np.abs(yn - y)))
        y = yn

        if er < tol:
            d, m, mu = y
            if d < 1.0e-9:
                d = 0.0
            o = obs(bd, p, (d, m, mu))
            return Sol(float(abs(d)), float(m), float(mu),
                       o.na, o.nb, er, it)

    raise RuntimeError(f"SCF loop failed after {nmax} iterations")


def follow(xa: Iterable[float], fn: Callable[[float, Arr], Sol],
           y0: ArrayLike) -> tuple[Arr, list[Sol]]:
    """Follow a solution branch by continuation in any scalar parameter."""

    xx = np.asarray(list(xa), dtype=float)
    y = np.asarray(y0, dtype=float)
    out: list[Sol] = []
    for x in xx:
        s = fn(float(x), y)
        out.append(s)
        y = s.y
    return xx, out