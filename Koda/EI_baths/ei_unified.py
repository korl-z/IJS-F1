from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tqdm.auto import tqdm, trange

Arr = NDArray[np.float64]
GamFn = Callable[["MFState", Arr], Any]
CurFn = Callable[[Arr, Any, "MFState"], Arr]


@dataclass(frozen=True)
class Bands:
    """Two bare bands on a weighted momentum grid."""

    k: Arr
    ea: Arr
    eb: Arr
    w: Arr
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        k = np.asarray(self.k, dtype=float)
        ea = np.asarray(self.ea, dtype=float).reshape(-1)
        eb = np.asarray(self.eb, dtype=float).reshape(-1)
        w = np.asarray(self.w, dtype=float).reshape(-1)

        if k.ndim == 1:
            k = k[:, None]
        if k.ndim != 2:
            raise ValueError("k must have shape (N, dim)")
        if ea.shape != eb.shape or ea.shape != w.shape:
            raise ValueError("ea, eb, and w must have shape (N,)")
        if k.shape[0] != ea.size:
            raise ValueError("k and the energy arrays have different sizes")
        if np.prod(self.shape, dtype=int) != ea.size:
            raise ValueError("shape is inconsistent with the number of modes")
        if np.any(~np.isfinite(k)) or np.any(~np.isfinite(ea + eb + w)):
            raise ValueError("band data must be finite")
        if np.any(w < 0.0) or np.sum(w) <= 0.0:
            raise ValueError("weights must be nonnegative with positive sum")

        w = w / np.sum(w)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "ea", ea)
        object.__setattr__(self, "eb", eb)
        object.__setattr__(self, "w", w)

    @property
    def size(self) -> int:
        return self.ea.size

    @property
    def dim(self) -> int:
        return self.k.shape[1]


@dataclass(frozen=True)
class MFPars:
    """Mean-field parameters: V, na+nb, U_H=V, ce none, sicer custom"""

    v: float
    n: float = 1.0
    uh: float | None = None

    @property
    def h(self) -> float:
        return self.v if self.uh is None else float(self.uh)


@dataclass(frozen=True)
class MFState:
    """Diagonalized mean-field state."""

    bd: Bands
    p: MFPars
    d: float
    m: float
    na: float
    nb: float
    eah: Arr
    ebh: Arr
    e: Arr
    u: Arr
    v: Arr


@dataclass(frozen=True)
class FDState:
    n: Arr
    mu: float


@dataclass(frozen=True)
class Targets:
    d: float
    m: float
    n: float
    na: float
    nb: float


@dataclass(frozen=True)
class EqSol:
    d: float
    m: float
    mu: float
    n: Arr
    st: MFState
    err: float
    it: int
    ok: bool
    hist: dict[str, Arr]


@dataclass(frozen=True)
class NormalSol:
    m: float
    mu: float
    n: Arr
    st: MFState
    err: float
    it: int
    ok: bool


@dataclass(frozen=True)
class OpenSol:
    d: float
    m: float
    n: Arr
    eta: Arr
    st: MFState
    cur: Arr
    err: float
    it: int
    time: float
    ok: bool
    hist: dict[str, Arr]


def tb_1d(nk: int, gap: float, ta: float, tb: float, a0: float = 1.0) -> Bands:
    """Nearest-neighbor one-dimensional tight-binding bands."""
    k = np.linspace(-np.pi / a0, np.pi / a0, nk, endpoint=False)
    c = np.cos(k * a0)
    ea = 0.5 * gap - 2.0 * ta * c
    eb = -0.5 * gap - 2.0 * tb * c
    return Bands(k[:, None], ea, eb, np.ones(nk), (nk,))


def tb_2d(nk: int, gap: float, ta: float, tb: float, a0: float = 1.0) -> Bands:
    """Nearest-neighbor square-lattice tight-binding bands."""
    q = np.linspace(-np.pi / a0, np.pi / a0, nk, endpoint=False)
    kx, ky = np.meshgrid(q, q, indexing="ij")
    c = np.cos(kx * a0) + np.cos(ky * a0)
    ea = 0.5 * gap - 2.0 * ta * c
    eb = -0.5 * gap - 2.0 * tb * c
    k = np.stack((kx, ky), axis=-1).reshape(-1, 2)
    return Bands(k, ea.reshape(-1), eb.reshape(-1), np.ones(nk * nk), (nk, nk))


def free_1d(
    nk: int, kmax: float, gap: float, ma: float = 1.0, mb: float = 1.0
) -> Bands:
    """One-dimensional electron and hole parabolic bands."""
    k = np.linspace(-kmax, kmax, nk, endpoint=False)
    ea = 0.5 * gap + k * k / (2.0 * ma)
    eb = -0.5 * gap - k * k / (2.0 * mb)
    return Bands(k[:, None], ea, eb, np.ones(nk), (nk,))


def reshape_mode(a: ArrayLike, bd: Bands) -> Arr:
    """Restore the original grid shape on the last array axis."""
    a = np.asarray(a, dtype=float)
    return a.reshape(a.shape[:-1] + bd.shape)


def mf_state(bd: Bands, p: MFPars, d: float, m: float) -> MFState:
    """Build and diagonalize the Hartree-shifted EI Hamiltonian."""
    na = 0.5 * (p.n + m)
    nb = 0.5 * (p.n - m)
    eah = bd.ea + p.h * nb
    ebh = bd.eb + p.h * na
    av = 0.5 * (eah + ebh)
    xi = 0.5 * (eah - ebh)
    ek = np.maximum(np.sqrt(xi * xi + d * d), 1.0e-14)
    z = np.clip(xi / ek, -1.0, 1.0)
    u = np.sqrt(0.5 * (1.0 + z))
    v = np.copysign(np.sqrt(0.5 * (1.0 - z)), d)
    e = np.stack((av + ek, av - ek))
    return MFState(bd, p, float(d), float(m), na, nb, eah, ebh, e, u, v)


def _fill(n: Arr, w: Arr) -> float:
    return float(np.sum(w * np.sum(n, axis=0)))


def fermi(
    st: MFState, t: float, tol: float = 1.0e-13, nmax: int = 120, prog: bool = False
) -> FDState:
    """Return the weighted FD distribution and its chemical potential."""
    n0 = st.p.n

    es = max(
        np.ptp(st.e), t, 1.0
    )  # kostruira ptp energijsko skalo, za interval za iskanje mu
    lo = float(st.e.min() - 50.0 * es)  # meje za mu
    hi = float(st.e.max() + 50.0 * es)  # meje za mu
    mu = 0.5 * (lo + hi)  # init guess

    for _ in trange(nmax, desc="FD kemijski potencial", disable=not prog):
        mu = 0.5 * (lo + hi)
        z = np.clip(
            (st.e - mu) / t, -500.0, 500.0
        )  # clipped (energija(diagonalizirano)-mu) / T
        n = 1.0 / (np.exp(z) + 1.0)
        q = _fill(
            n, st.bd.w
        )  # zracuna current filing sum_k (n_alpha,k + n_beta,k) * w_k
        if q < n0:
            lo = mu
        else:
            hi = mu
        if hi - lo <= tol * max(1.0, abs(mu)):
            break  # bisekcija

    mu = 0.5 * (lo + hi)
    z = np.clip((st.e - mu) / t, -500.0, 500.0)
    n = 1.0 / (np.exp(z) + 1.0)
    return FDState(n, float(mu))


def fermi_zero(st: MFState) -> FDState:
    """Return an exact zero-temperature filling on the discrete grid."""
    n0 = st.p.n
    y = st.e.reshape(-1)
    w = np.tile(st.bd.w, 2)
    ix = np.argsort(y)
    cw = np.cumsum(w[ix])
    n = np.zeros_like(y)

    if n0 <= 0.0:
        return FDState(n.reshape(st.e.shape), float(y.min() - 1.0))
    if n0 >= 2.0:
        n.fill(1.0)
        return FDState(n.reshape(st.e.shape), float(y.max() + 1.0))

    j = int(np.searchsorted(cw, n0, side="left"))
    n[ix[:j]] = 1.0
    q0 = 0.0 if j == 0 else cw[j - 1]
    n[ix[j]] = np.clip((n0 - q0) / w[ix[j]], 0.0, 1.0)

    if n[ix[j]] < 1.0 or j + 1 >= y.size:
        mu = y[ix[j]]
    else:
        mu = 0.5 * (y[ix[j]] + y[ix[j + 1]])
    return FDState(n.reshape(st.e.shape), float(mu))


def band_occ(st: MFState, n: ArrayLike) -> tuple[Arr, Arr]:
    """Transform diagonal quasiparticle occupations to the a and b bands."""
    n = np.asarray(n, dtype=float)
    u2 = st.u * st.u
    v2 = st.v * st.v
    na = u2 * n[0] + v2 * n[1]
    nb = v2 * n[0] + u2 * n[1]
    return na, nb


def targets(st: MFState, n: ArrayLike) -> Targets:
    """Return weighted gap, imbalance, filling, and band occupations."""
    n = np.asarray(n, dtype=float)
    na, nb = band_occ(st, n)
    w = st.bd.w
    d = st.p.v * np.sum(w * st.u * st.v * (n[1] - n[0]))
    ma = np.sum(w * na)
    mb = np.sum(w * nb)
    return Targets(float(d), float(ma - mb), _fill(n, w), float(ma), float(mb))


def solve_eq(
    bd: Bands,
    p: MFPars,
    t: float,
    d: float = 0.2,
    m: float = 0.0,
    mix: tuple[float, float] = (0.2, 0.2),
    tol: float = 1.0e-10,
    nmax: int = 20000,
    chk: int = 20,
    prog: bool = True,
) -> EqSol:
    """Solve the weighted equilibrium EI fixed-point equations."""
    ad, am = map(float, mix)  # mixture update parameteri
    hs = {"it": [], "err": [], "d": [], "m": [], "mu": []}  # history slovar
    ok = False
    it = 0
    bar = trange(1, nmax + 1, desc="Equilibrium EI", disable=not prog)

    for it in bar:
        st = mf_state(bd, p, d, m)
        fs = fermi_zero(st) if t <= 0.0 else fermi(st, t)
        tg = targets(st, fs.n)
        rd = tg.d - d
        rm = tg.m - m
        er = max(abs(rd), abs(rm), abs(tg.n - p.n))

        d += ad * rd
        m += am * rm

        if it == 1 or it % chk == 0 or er < tol:
            hs["it"].append(it)
            hs["err"].append(er)
            hs["d"].append(d)
            hs["m"].append(m)
            hs["mu"].append(fs.mu)
            if prog:
                bar.set_postfix(err=f"{er:.2e}", d=f"{d:.5f}", m=f"{m:.5f}")
        if er < tol:
            ok = True
            break

    st = mf_state(bd, p, d, m)
    fs = fermi_zero(st) if t <= 0.0 else fermi(st, t)
    tg = targets(st, fs.n)
    er = max(abs(tg.d - d), abs(tg.m - m), abs(tg.n - p.n))
    hh = {q: np.asarray(z) for q, z in hs.items()}
    return EqSol(
        float(abs(d)), float(m), fs.mu, fs.n, st, float(er), it, ok or er < tol, hh
    )


def solve_normal(
    bd: Bands,
    p: MFPars,
    t: float,
    m: float = 0.0,
    mix: float = 0.3,
    tol: float = 1.0e-11,
    nmax: int = 10000,
    prog: bool = False,
) -> NormalSol:
    """Solve the normal-state polarization at fixed filling."""
    ok = False
    it = 0
    bar = trange(1, nmax + 1, desc="Normal state", disable=not prog)
    for it in bar:
        st = mf_state(bd, p, 0.0, m)  # fix delta=0
        fs = fermi(st, t)
        tg = targets(st, fs.n)
        er = max(abs(tg.m - m), abs(tg.n - p.n))
        m += mix * (tg.m - m)
        if er < tol:
            ok = True
            break

    st = mf_state(bd, p, 0.0, m)
    fs = fermi(st, t)
    tg = targets(st, fs.n)
    er = max(abs(tg.m - m), abs(tg.n - p.n))
    return NormalSol(float(m), fs.mu, fs.n, st, float(er), it, ok or er < tol)


def pair_lambda(
    bd: Bands, p: MFPars, t: float, m: float = 0.0, tol: float = 1.0e-11
) -> tuple[float, NormalSol]:
    """Return the normal-state linearized pairing eigenvalue."""
    s = solve_normal(bd, p, t, m=m, tol=tol)
    ek = 0.5 * (s.st.e[0] - s.st.e[1])
    q = np.empty_like(ek)
    ix = ek > 1.0e-12
    q[ix] = (s.n[1, ix] - s.n[0, ix]) / (2.0 * ek[ix])
    f0 = 0.5 * (s.n[0, ~ix] + s.n[1, ~ix])
    q[~ix] = f0 * (1.0 - f0) / t
    la = p.v * np.sum(bd.w * q)  # λ​=V ∑​wk (​nβk​−nαk​​) / ​2Ek.
    return float(la), s


def critical_temperature(
    bd: Bands,
    p: MFPars,
    tlo: float = 1.0e-4,
    thi: float = 5.0,
    tol: float = 1.0e-8,
    nmax: int = 80,
    prog: bool = True,
) -> float:
    """Find a continuous transition from the linearized gap equation."""
    ll, sl = pair_lambda(bd, p, tlo)
    lh, sh = pair_lambda(bd, p, thi)
    if ll <= 1.0:
        raise ValueError("ni nestabilnosti")
    if lh >= 1.0:
        raise ValueError("povecaj thi")

    ml = sl.m
    mh = sh.m
    for _ in trange(nmax, desc="kriticna temp bisekcija", disable=not prog):
        tm = 0.5 * (tlo + thi)
        lm, sm = pair_lambda(bd, p, tm, m=0.5 * (ml + mh))
        if lm >= 1.0:
            tlo, ml = tm, sm.m
        else:
            thi, mh = tm, sm.m
        if thi - tlo <= tol * max(1.0, tm):
            return 0.5 * (tlo + thi)
    raise RuntimeError("Tc did not converge")


def gam_db(
    t: float, kap: float = 1.0, wc: float = np.inf, orb: str | None = None
) -> GamFn:
    """Make a bounded dense transfer rate with detailed balance."""

    def gam(st: MFState, n: Arr) -> Arr:
        de = st.e[:, :, None, None] - st.e[None, None, :, :]
        if np.isinf(wc):
            sp = 1.0
        else:
            sp = np.exp(-((de / wc) ** 2))
        r = kap * sp * np.exp(-np.logaddexp(0.0, de / t))

        if orb is not None:
            q = np.stack((st.u, st.v)) if orb == "a" else np.stack((-st.v, st.u))
            a = q[:, :, None, None] * q[None, None, :, :]
            r = r * a * a

        rf = r.reshape(2 * st.bd.size, 2 * st.bd.size)
        np.fill_diagonal(rf, 0.0)
        return rf.reshape(2, st.bd.size, 2, st.bd.size)

    return gam


def gam_scalar(
    fn: Callable[[int, int, int, int, MFState, Arr], float], prog: bool = False
) -> GamFn:
    """Adapt a scalar rate function to the dense transfer tensor."""

    def gam(st: MFState, n: Arr) -> Arr:
        sh = (2, st.bd.size, 2, st.bd.size)
        r = np.empty(sh, dtype=float)
        it = np.ndindex(sh)
        for a, k, b, q in tqdm(
            it, total=int(np.prod(sh)), desc="Scalar rates", disable=not prog
        ):
            r[a, k, b, q] = fn(a, k, b, q, st, n)
        return r

    return gam


def dense_current(n: Arr, r: ArrayLike, st: MFState) -> Arr:
    """Pauli-blocked current for number-conserving transfer jumps."""
    n = np.asarray(n, dtype=float)
    r = np.asarray(r, dtype=float)

    y = n.reshape(-1)
    w = np.tile(st.bd.w, 2)
    rf = r.reshape(2 * st.bd.size, 2 * st.bd.size)
    gain = (1.0 - y) * (rf @ (w * y))
    loss = y * (rf.T @ (w * (1.0 - y)))
    return (gain - loss).reshape(n.shape)


@dataclass(frozen=True)
class Dissipator:
    """A compatible rate builder and kinetic current law."""

    gam: GamFn
    cur: CurFn = dense_current
    name: str = "bath"

    def current(self, st: MFState, n: Arr) -> Arr:
        return np.asarray(self.cur(n, self.gam(st, n), st), dtype=float)


def total_current(ds: Iterable[Dissipator], st: MFState, n: Arr) -> Arr:
    """Add currents from independent dissipators."""
    out = np.zeros_like(n, dtype=float)
    for d in ds:
        q = d.current(st, n)
        out += q
    return out


def number_rate(dn: ArrayLike, bd: Bands) -> float:
    """Return the weighted total particle-number rate."""
    dn = np.asarray(dn, dtype=float)
    return float(np.sum(bd.w * np.sum(dn, axis=0)))


def check_db(r: ArrayLike, st: MFState, t: float, eps: float = 1.0e-250) -> float:
    """Return the largest local detailed-balance log residual."""
    r = np.asarray(r, dtype=float).reshape(2 * st.bd.size, 2 * st.bd.size)
    e = st.e.reshape(-1)
    de = e[:, None] - e[None, :]
    ma = (r > eps) & (r.T > eps)
    np.fill_diagonal(ma, False)
    if not np.any(ma):
        return float("nan")
    z = np.log(r[ma]) - np.log(r.T[ma]) + de[ma] / t
    return float(np.max(np.abs(z)))


def lim_step(n: ArrayLike, dn: ArrayLike, dt: float, fac: float = 0.8) -> float:
    """Limit an Euler step so all occupations stay in the unit interval."""
    n = np.asarray(n, dtype=float)
    dn = np.asarray(dn, dtype=float)
    z = []
    jp = dn > 0.0
    jm = dn < 0.0
    if np.any(jp):
        z.append(np.min((1.0 - n[jp]) / dn[jp]))
    if np.any(jm):
        z.append(np.min(-n[jm] / dn[jm]))
    if z:
        dt = min(dt, fac * max(0.0, min(z)))
    return float(dt)


def solve_open(
    bd: Bands,
    p: MFPars,
    n: ArrayLike,
    ds: Iterable[Dissipator],
    d: float = 0.2,
    m: float = 0.0,
    dt: float = 0.1,
    td: float = 0.5,
    tm: float = 0.5,
    tol: float = 1.0e-8,
    nmax: int = 50000,
    chk: int = 10,
    prog: bool = True,
) -> OpenSol:
    """Relax occupations and mean fields to a coupled steady state."""
    n = np.asarray(n, dtype=float).copy()
    ds = tuple(ds)

    hs = {q: [] for q in ("it", "t", "err", "cur", "ed", "em", "d", "m", "n0")}
    tt = 0.0
    ok = False
    er = np.inf
    it = 0
    bar = trange(1, nmax + 1, desc="Open EI", disable=not prog)

    for it in bar:
        st = mf_state(bd, p, d, m)
        dn = total_current(ds, st, n)
        h = lim_step(n, dn, dt)
        if h <= 0.0:
            raise RuntimeError("korak kolapsiral na 0")
        n = np.clip(n + h * dn, 0.0, 1.0)  # current update
        tg = targets(st, n)  # targets update (d, m, n)
        zd = -np.expm1(-h / td)  # update koraki
        zm = -np.expm1(-h / tm)  # update koraki
        d += zd * (tg.d - d)  # zΔ​=1−e−h/td​,
        m += zm * (tg.m - m)  # zm​=1−e−h/tm
        tt += h  # time tracker

        if it == 1 or it % chk == 0 or it == nmax:  # zgodovina update
            stc = mf_state(bd, p, d, m)
            dnc = total_current(ds, stc, n)
            tgc = targets(stc, n)
            ei = float(np.max(np.abs(dnc)))
            ed = float(abs(tgc.d - d))
            em = float(abs(tgc.m - m))
            en = float(abs(tgc.n - p.n))
            er = max(ei, ed, em, en)
            hs["it"].append(it)
            hs["t"].append(tt)
            hs["err"].append(er)
            hs["cur"].append(ei)
            hs["ed"].append(ed)
            hs["em"].append(em)
            hs["d"].append(d)
            hs["m"].append(m)
            hs["n0"].append(tgc.n)
            if prog:
                bar.set_postfix(err=f"{er:.2e}", d=f"{d:.5f}", m=f"{m:.5f}")
            if er < tol:
                ok = True
                break

    st = mf_state(bd, p, d, m)
    dn = total_current(ds, st, n)
    tg = targets(st, n)
    er = max(float(np.max(np.abs(dn))), abs(tg.d - d), abs(tg.m - m), abs(tg.n - p.n))
    nc = np.clip(n, 1.0e-14, 1.0 - 1.0e-14)
    eta = np.log((1.0 - nc) / nc)
    hh = {q: np.asarray(z) for q, z in hs.items()}
    return OpenSol(
        float(d), float(m), n, eta, st, dn, float(er), it, tt, ok or er < tol, hh
    )


__all__ = [
    "Bands",
    "MFPars",
    "MFState",
    "FDState",
    "Targets",
    "EqSol",
    "NormalSol",
    "OpenSol",
    "Dissipator",
    "from_fn",
    "tb_1d",
    "tb_2d",
    "free_1d",
    "reshape_mode",
    "mf_state",
    "fermi",
    "fermi_zero",
    "band_occ",
    "targets",
    "solve_eq",
    "solve_normal",
    "pair_lambda",
    "critical_temperature",
    "gam_db",
    "gam_scalar",
    "dense_current",
    "total_current",
    "number_rate",
    "check_db",
    "lim_step",
    "solve_open",
]
