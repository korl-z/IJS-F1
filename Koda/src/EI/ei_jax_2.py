from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

import EI.ei_unified as eu
import EI.ei_phonon as pb

F32 = jnp.float64
I32 = jnp.int64
RateFn = Callable[[Any, Any, Any, Any, Any], Any]


class JState(NamedTuple):
    """state vektor za MF, jax verzija"""

    d: Any
    m: Any
    na: Any
    nb: Any
    eah: Any
    ebh: Any
    e: Any
    u: Any
    v: Any


class JTargets(NamedTuple):
    """update targets za self consistent loop"""

    d: Any
    m: Any
    n: Any
    na: Any
    nb: Any


def _as_jstate(st):
    """Accept returned MFState objects at the public JAX helper boundary.

    No energy shift is applied here: the supplied state must already use the
    desired frame. States returned by this module use -p.h/2 throughout.
    """
    return JState(*(jnp.asarray(getattr(st, q), dtype=F32) for q in JState._fields))


def rate_db(ei: Any, ej: Any, de: Any, t: Any, mu: Any) -> Any:
    """originalni rate za gamma_ij"""
    return jax.nn.sigmoid(-de / t)


def rate_product(ei: Any, ej: Any, de: Any, t: Any, mu: Any) -> Any:
    """product rate iz clanka, dodan mu, ker imamo Hartree shifte"""
    fi = jax.nn.sigmoid((mu - ei) / t)
    hj = jax.nn.sigmoid((ej - mu) / t)
    return fi * hj


def _kap_mat(kap):
    """"""
    a = np.asarray(kap, dtype=float)
    if a.ndim == 0:
        z = float(a)
        a = np.full((2, 2), z, dtype=float)
    elif a.shape == (3,):
        kaa, kab, kbb = map(float, a)
        a = np.array([[kaa, kab], [kab, kbb]], dtype=float)
    if a.shape != (2, 2) or np.any(~np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("kap must be a nonnegative scalar, length 3, or 2x2 array")
    return a


@dataclass(frozen=True)
class Bath:
    """Parameters and rate factor of one transfer bath."""

    t: float
    kap: Any = 1.0
    wc: float = np.inf
    orb: str | None = None
    name: str = "bath"
    mu: float = 0.0
    rate: RateFn = rate_product

    def __post_init__(self) -> None:
        if self.t <= 0.0 or np.isnan(self.t) or self.wc <= 0.0 or np.isnan(self.wc):
            raise ValueError("t and wc must be positive, and kap nonnegative")
        if not np.isfinite(self.mu):
            raise ValueError("mu must be finite and in the shifted energy frame")
        if self.orb not in (None, "a", "b"):
            raise ValueError("orb must be None, 'a', or 'b'")
        a = _kap_mat(self.kap)
        q = tuple(tuple(float(x) for x in row) for row in a)
        object.__setattr__(self, "kap", q)


def gam_db(
    t: float,
    kap: Any = 1.0,
    wc: float = np.inf,
    orb: str | None = None,
    name: str = "bath",
    mu: float = 0.0,
    rate: RateFn = rate_product,
) -> Bath:
    """Return a bath using the selected JAX rate factor."""
    return Bath(t=float(t), kap=kap, wc=float(wc), orb=orb, name=name, mu=float(mu), rate=rate)


def _pars(bd: eu.Bands, p: eu.MFPars) -> tuple[Any, ...]:
    ea = jnp.asarray(bd.ea, dtype=F32)
    eb = jnp.asarray(bd.eb, dtype=F32)
    w = jnp.asarray(bd.w, dtype=F32)
    pv = jnp.asarray(p.v, dtype=F32)
    pn = jnp.asarray(p.n, dtype=F32)
    ph = jnp.asarray(p.h, dtype=F32)
    return ea, eb, w, pv, pn, ph


def _baths(bs):
    bs = tuple(bs)
    if not bs:
        raise ValueError("at least one bath is required")
    if any(isinstance(b, pb.PhononBath) for b in bs):
        if not all(isinstance(b, pb.PhononBath) for b in bs):
            raise TypeError("use phonon baths together, without placeholder baths")
        return pb.pack(bs), ()
    oc = {None: 0, "a": 1, "b": 2}
    bt = jnp.asarray([b.t for b in bs], dtype=F32)
    bk = jnp.asarray(np.stack([_kap_mat(b.kap) for b in bs]), dtype=F32)
    bw = jnp.asarray([b.wc for b in bs], dtype=F32)
    bo = jnp.asarray([oc[b.orb] for b in bs], dtype=I32)
    bm = jnp.asarray([b.mu for b in bs], dtype=F32)
    rf = tuple(b.rate for b in bs)
    return (bt, bk, bw, bo, bm), rf


def _mf(ea: Any, eb: Any, pn: Any, ph: Any, d: Any, m: Any) -> JState:
    na = F32(0.5) * (pn + m)
    nb = F32(0.5) * (pn - m)
    # Shift the entire Hamiltonian, so all stored energies share one frame.
    sh = F32(0.5) * ph
    eah = ea + ph * nb - sh
    ebh = eb + ph * na - sh
    av = F32(0.5) * (eah + ebh)
    xi = F32(0.5) * (eah - ebh)
    ek = jnp.hypot(xi, d)
    safe = jnp.where(ek > F32(0.0), ek, F32(1.0))
    large = jnp.sqrt((safe + jnp.abs(xi)) / (F32(2.0) * safe))
    small = jnp.abs(d) / (F32(2.0) * safe * large)
    u = jnp.where(xi >= F32(0.0), large, small)
    v = jnp.copysign(jnp.where(xi >= F32(0.0), small, large), d)
    u = jnp.where(ek > F32(0.0), u, F32(1.0))
    v = jnp.where(ek > F32(0.0), v, F32(0.0))
    e = jnp.stack((av + ek, av - ek))
    return JState(d, m, na, nb, eah, ebh, e, u, v)


@jax.jit
def _mf_jit(ea: Any, eb: Any, pn: Any, ph: Any, d: Any, m: Any) -> JState:
    return _mf(ea, eb, pn, ph, d, m)


def mf_state(bd: eu.Bands, p: eu.MFPars, d: float, m: float) -> JState:
    """Build a compatible mean-field state on the JAX device."""
    ea, eb, _, _, pn, ph = _pars(bd, p)
    return _mf_jit(ea, eb, pn, ph, F32(d), F32(m))


def _band_occ(st: JState, n: Any) -> tuple[Any, Any]:
    u2 = st.u * st.u
    v2 = st.v * st.v
    na = u2 * n[0] + v2 * n[1]
    nb = v2 * n[0] + u2 * n[1]
    return na, nb


_band_occ_jit = jax.jit(_band_occ)


def band_occ(st: JState, n: Any) -> tuple[Any, Any]:
    """Rotate quasiparticle occupations back to the bare bands."""
    st = _as_jstate(st)
    n = jnp.asarray(n, dtype=F32)
    if n.shape != st.e.shape:
        raise ValueError("n must have shape (2, N)")
    return _band_occ_jit(st, n)


def _targets(st: JState, n: Any, w: Any, pv: Any) -> JTargets:
    na, nb = _band_occ(st, n)
    d = pv * jnp.sum(w * st.u * st.v * (n[1] - n[0]))
    ma = jnp.sum(w * na)
    mb = jnp.sum(w * nb)
    nf = jnp.sum(w * jnp.sum(n, axis=0))
    return JTargets(d, ma - mb, nf, ma, mb)


@jax.jit
def _targets_jit(st: JState, n: Any, w: Any, pv: Any) -> JTargets:
    return _targets(st, n, w, pv)


def targets(bd: eu.Bands, p: eu.MFPars, st: JState, n: Any) -> JTargets:
    """Return the device-side gap, imbalance, and filling targets."""
    st = _as_jstate(st)
    n = jnp.asarray(n, dtype=F32)
    if n.shape != st.e.shape:
        raise ValueError("n must have shape (2, N)")
    _, _, w, pv, _, _ = _pars(bd, p)
    return _targets_jit(st, n, w, pv)


def _qv(st: JState) -> tuple[Any, Any]:
    qa = jnp.concatenate((st.u, st.v))
    qb = jnp.concatenate((-st.v, st.u))
    return qa, qb


def _af(qa: Any, qb: Any, o: Any) -> Any:
    q = jnp.where(o == 1, qa, jnp.where(o == 2, qb, jnp.ones_like(qa)))
    a = q[:, None] * q[None, :]
    return jnp.where(o == 0, jnp.ones_like(a), a * a)


def _rates(st: JState, bp: tuple[Any, ...], rf: tuple[RateFn, ...]) -> Any:
    if isinstance(bp, pb.Pack):
        return pb.rates(st, bp)
    bt, bk, bw, bo, bm = bp
    e = st.e.reshape(-1)
    ei = e[:, None]
    ej = e[None, :]
    de = ei - ej
    la = jnp.repeat(jnp.arange(2, dtype=I32), st.e.shape[1])
    qa, qb = _qv(st)
    r = jnp.zeros_like(de)
    for i, fn in enumerate(rf):
        sp = jnp.exp(-jnp.square(de / bw[i]))
        af = _af(qa, qb, bo[i])
        km = bk[i, la[:, None], la[None, :]]
        r = r + km * sp * fn(ei, ej, de, bt[i], bm[i]) * af
    ma = F32(1.0) - jnp.eye(e.size, dtype=F32)
    return r * ma


@partial(jax.jit, static_argnames=("rf",))
def _rates_jit(st: JState, bp: tuple[Any, ...], rf: tuple[RateFn, ...]) -> Any:
    return _rates(st, bp, rf)


def dense_rates(st: JState, bs: Iterable[Bath]) -> Any:
    """Build the summed dense rate matrix on the JAX device."""
    st = _as_jstate(st)
    bp, rf = _baths(bs)
    return _rates_jit(st, bp, rf)


def _cur_dense(st: JState, n: Any, w: Any, bp: tuple[Any, ...], rf: tuple[RateFn, ...]) -> Any:
    return _cur_mat(n, w, _rates(st, bp, rf))


def _ab_mat(n: Any, w: Any, r: Any) -> tuple[Any, Any]:
    y = n.reshape(-1)
    ww = jnp.tile(w, 2)
    a = r @ (ww * y)
    b = r.T @ (ww * (F32(1.0) - y))
    return a.reshape(n.shape), b.reshape(n.shape)


def _cur_mat(n: Any, w: Any, r: Any) -> Any:
    a, b = _ab_mat(n, w, r)
    return (F32(1.0) - n) * a - n * b


def _ab_dense(
    st: Any, n: Any, w: Any, bp: tuple[Any, ...], rf: tuple[RateFn, ...]
) -> tuple[Any, Any]:
    return _ab_mat(n, w, _rates(st, bp, rf))


def _cur_block(
    st: JState, n: Any, w: Any, bp: tuple[Any, ...], rf: tuple[RateFn, ...], block: int
) -> Any:
    if isinstance(bp, pb.Pack):
        return pb.cur_block(st, n, w, bp, block)
    bt, bk, bw, bo, bm = bp
    y = n.reshape(-1)
    e = st.e.reshape(-1)
    ww = jnp.tile(w, 2)
    qa, qb = _qv(st)
    nm = e.size
    la = jnp.repeat(jnp.arange(2, dtype=I32), st.e.shape[1])
    nb = (nm + block - 1) // block
    npad = nb * block - nm
    lp = jnp.pad(la, (0, npad))
    ep = jnp.pad(e, (0, npad))
    yp = jnp.pad(y, (0, npad))
    wp = jnp.pad(ww, (0, npad))
    ap = jnp.pad(qa, (0, npad))
    bpv = jnp.pad(qb, (0, npad))
    ii = jnp.arange(nm, dtype=I32)[:, None]

    def src_body(ib: int, gl: tuple[Any, Any]) -> tuple[Any, Any]:
        ga, lo = gl
        j0 = ib * block
        ej = jax.lax.dynamic_slice_in_dim(ep, j0, block)[None, :]
        yj = jax.lax.dynamic_slice_in_dim(yp, j0, block)
        wj = jax.lax.dynamic_slice_in_dim(wp, j0, block)
        aj = jax.lax.dynamic_slice_in_dim(ap, j0, block)
        bj = jax.lax.dynamic_slice_in_dim(bpv, j0, block)
        lj = jax.lax.dynamic_slice_in_dim(lp, j0, block)
        jj = j0 + jnp.arange(block, dtype=I32)
        vm = jj < nm
        pm = vm[None, :] & (ii != jj[None, :])
        ei = e[:, None]
        de = ei - ej
        for ir, fn in enumerate(rf):
            qi = jnp.where(bo[ir] == 1, qa, jnp.where(bo[ir] == 2, qb, jnp.ones_like(qa)))
            qj = jnp.where(bo[ir] == 1, aj, jnp.where(bo[ir] == 2, bj, jnp.ones_like(aj)))
            aa = qj[None, :] * qi[:, None]
            af = jnp.where(bo[ir] == 0, jnp.ones_like(aa), aa * aa)
            kf = bk[ir, la[:, None], lj[None, :]]
            kr = bk[ir, lj[None, :], la[:, None]]
            sp = jnp.exp(-jnp.square(de / bw[ir]))
            cf = kf * sp * af * pm
            cr = kr * sp * af * pm
            r1 = cf * fn(ei, ej, de, bt[ir], bm[ir])
            r2 = cr * fn(ej, ei, -de, bt[ir], bm[ir])
            ga = ga + jnp.sum(r1 * (wj * yj)[None, :], axis=1)
            lo = lo + jnp.sum(r2 * (wj * (F32(1.0) - yj))[None, :], axis=1)
        return ga, lo

    z = jnp.zeros_like(y)
    ga, lo = jax.lax.fori_loop(0, nb, src_body, (z, z))
    dn = (F32(1.0) - y) * ga - y * lo
    return dn.reshape(n.shape)


def _cur_pair(st: JState, n: Any, bp: tuple[Any, ...], rf: tuple[RateFn, ...]) -> Any:
    if isinstance(bp, pb.Pack):
        raise ValueError("phonon baths require mode='dense' or mode='block'")
    bt, bk, bw, bo, bm = bp
    ea = st.e[0]
    eb = st.e[1]
    de = ea - eb
    na = n[0]
    nb = n[1]
    uv = jnp.square(st.u * st.v)
    rab = jnp.zeros_like(de)
    rba = jnp.zeros_like(de)
    for ir, fn in enumerate(rf):
        sp = jnp.exp(-jnp.square(de / bw[ir]))
        af = jnp.where(bo[ir] == 0, jnp.ones_like(de), uv)
        cf = bk[ir, 0, 1] * sp * af
        cr = bk[ir, 1, 0] * sp * af
        rab = rab + cf * fn(ea, eb, de, bt[ir], bm[ir])
        rba = rba + cr * fn(eb, ea, -de, bt[ir], bm[ir])
    q = (F32(1.0) - na) * rab * nb - na * rba * (F32(1.0) - nb)
    return jnp.stack((q, -q))


def _cur(
    st: JState, n: Any, w: Any, bp: tuple[Any, ...], rf: tuple[RateFn, ...], mode: str, block: int
) -> Any:
    if mode == "pair":
        return _cur_pair(st, n, bp, rf)
    if mode == "dense":
        return _cur_dense(st, n, w, bp, rf)
    if mode == "block":
        return _cur_block(st, n, w, bp, rf, block)
    raise ValueError("mode must be 'pair', 'dense', or 'block'")


@partial(jax.jit, static_argnames=("rf", "mode", "block"))
def _cur_jit(
    st: JState, n: Any, w: Any, bp: tuple[Any, ...], rf: tuple[RateFn, ...], mode: str, block: int
) -> Any:
    return _cur(st, n, w, bp, rf, mode, block)


def total_current(
    bd: eu.Bands, st: JState, n: Any, bs: Iterable[Bath], mode: str = "dense", block: int = 512
) -> Any:
    """Evaluate the summed bath current on the JAX device."""
    if block <= 0:
        raise ValueError("block must be positive")
    st = _as_jstate(st)
    n = jnp.asarray(n, dtype=F32)
    if n.shape != st.e.shape:
        raise ValueError("n must have shape (2, N)")
    w = jnp.asarray(bd.w, dtype=F32)
    bp, rf = _baths(bs)
    return _cur_jit(st, n, w, bp, rf, mode=mode, block=block)


def dense_current(bd: eu.Bands, st: JState, n: Any, bs: Iterable[Bath]) -> Any:
    """Evaluate the current using a full rate matrix."""
    return total_current(bd, st, n, bs, mode="dense")


def blocked_current(bd: eu.Bands, st: JState, n: Any, bs: Iterable[Bath], block: int = 512) -> Any:
    """Evaluate the current using source blocks of the rate matrix."""
    return total_current(bd, st, n, bs, mode="block", block=block)


def _lim(n, dn, dt, fac=F32(0.8), eps=F32(0.0)):
    """Bound an Euler update. The physical solver always uses eps=0."""
    jp, jm = dn > 0.0, dn < 0.0
    dp = jnp.where(jp, dn, F32(1.0))
    dm = jnp.where(jm, dn, F32(-1.0))
    hp = jnp.min(jnp.where(jp, (F32(1.0) + eps - n) / dp, jnp.inf))
    hm = jnp.min(jnp.where(jm, (-eps - n) / dm, jnp.inf))
    hb = fac * jnp.maximum(F32(0.0), jnp.minimum(hp, hm))
    return jnp.minimum(dt, hb)


@jax.jit
def _lim_jit(n, dn, dt, fac, eps):
    return _lim(n, dn, dt, fac, eps)


def lim_step(n, dn, dt: float, fac: float = 0.8, eps: float = 0.0):
    """Return a bounded Euler step; eps expands the interval if nonzero."""
    if not np.isfinite(dt) or dt <= 0 or not 0 < fac <= 1:
        raise ValueError("dt must be finite and positive, with 0 < fac <= 1")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    return _lim_jit(
        jnp.asarray(n, dtype=F32), jnp.asarray(dn, dtype=F32), F32(dt), F32(fac), F32(eps)
    )


def _ab2(n, dn, dnp, hp, first, dt):
    """Variable-step AB2 with trial rejection and bounded Euler fallback.

    Return the new occupations, accepted interval, and Euler flag.
    This is a bounds controller, NOT an accuracy or stiffness controller.
    """
    # Limit growth relative to the preceding accepted interval.
    hc = jnp.where(first, dt, jnp.minimum(dt, F32(2.0) * hp))
    he = _lim(n, dn, hc, eps=F32(0.0))

    def trial(h):
        rr = h / jnp.where(hp > 0, hp, F32(1.0))
        ds = dn + F32(0.5) * rr * (dn - dnp)
        ds = jnp.where(first, dn, ds)
        return n + h * ds

    def valid(y):
        return jnp.all(jnp.isfinite(y)) & jnp.all(y >= F32(0.0)) & jnp.all(y <= F32(1.0))

    def cond(ca):
        h, y, j = ca
        return (~valid(y)) & (j < 16) & (h > 0)

    def body(ca):
        h, _, j = ca
        h = F32(0.5) * h
        return h, trial(h), j + 1

    h, y, _ = jax.lax.while_loop(cond, body, (he, trial(he), jnp.asarray(0, dtype=I32)))
    good = valid(y) & (h > 0)
    # If extrapolation points outward at an occupation boundary, use dn.
    ye = n + he * dn
    y = jnp.where(good, y, ye)
    h = jnp.where(good, h, he)
    # Do not hide failed steps by clipping or by projecting the filling.
    safe = valid(y) & jnp.isfinite(h) & (h > 0)
    h = jnp.where(safe, h, F32(0.0))
    y = jnp.where(safe, y, n)
    return y, h, first | (~good)


def _open_step(ca, _, ea, eb, w, pv, pn, ph, bp, rf, dt, td, tm, mode: str, block: int):
    n, d, m, tt, dnp, hp, first = ca
    st = _mf(ea, eb, pn, ph, d, m)
    dn = _cur(st, n, w, bp, rf, mode, block)
    n, h, _ = _ab2(n, dn, dnp, hp, first, dt)

    # Keep the original mean-field relaxation law.
    tg = _targets(st, n, w, pv)
    zd = -jnp.expm1(-h / td)
    zm = -jnp.expm1(-h / tm)
    d = d + zd * (tg.d - d)
    m = m + zm * (tg.m - m)
    return (n, d, m, tt + h, dn, h, jnp.asarray(False)), None


@partial(jax.jit, static_argnames=("rf", "ns", "mode", "block"))
def _open_chunk(
    ea, eb, w, pv, pn, ph, bp, rf,
    n, d, m, tt, dnp, hp, first, dt, td, tm,
    ns: int, mode: str, block: int,
):
    fn = partial(
        _open_step, ea=ea, eb=eb, w=w, pv=pv, pn=pn, ph=ph,
        bp=bp, rf=rf, dt=dt, td=td, tm=tm, mode=mode, block=block,
    )
    ca, _ = jax.lax.scan(fn, (n, d, m, tt, dnp, hp, first), xs=None, length=ns)
    return ca


def _fixed_step(ca, _, ea, eb, w, pv, pn, ph, r, xn, xd, xm):
    n, d, m = ca
    st = _mf(ea, eb, pn, ph, d, m)
    # r is frozen for the entire chunk, but a and b depend on the current n.
    a, b = _ab_mat(n, w, r)
    q = a + b
    qcut = F32(1.0e-12) * jnp.max(q)
    active = q > qcut
    nt = jnp.where(active, a / jnp.where(active, q, F32(1.0)), n)
    n = _fix_fill(n + xn * (nt - n), w, pn)
    tg = _targets(st, n, w, pv)
    d = d + xd * (tg.d - d)
    m = m + xm * (tg.m - m)
    return (n, d, m), None


@partial(jax.jit, static_argnames=("ns",))
def _fixed_chunk(ea, eb, w, pv, pn, ph, r, n, d, m, xn, xd, xm, ns: int):
    fn = partial(_fixed_step, ea=ea, eb=eb, w=w, pv=pv, pn=pn, ph=ph, r=r, xn=xn, xd=xd, xm=xm)
    ca, _ = jax.lax.scan(fn, (n, d, m), xs=None, length=ns)
    return ca


def _diag_vals(st, n, w, pv, pn, d, m, dn):
    tg = _targets(st, n, w, pv)
    ec = jnp.max(jnp.abs(dn))
    ed = jnp.abs(tg.d - d)
    em = jnp.abs(tg.m - m)
    en = jnp.abs(tg.n - pn)
    er = jnp.maximum(jnp.maximum(ec, ed), jnp.maximum(em, en))
    return dn, er, ec, ed, em, tg.n


@partial(jax.jit, static_argnames=("rf", "mode", "block"))
def _open_diag(ea, eb, w, pv, pn, ph, bp, rf, n, d, m, mode: str, block: int):
    st = _mf(ea, eb, pn, ph, d, m)
    dn = _cur(st, n, w, bp, rf, mode, block)
    return _diag_vals(st, n, w, pv, pn, d, m, dn)


@jax.jit
def _fixed_diag(ea, eb, w, pv, pn, ph, r, n, d, m):
    # Caller must refresh r at the current (d, m) BEFORE calling this.
    st = _mf(ea, eb, pn, ph, d, m)
    return _diag_vals(st, n, w, pv, pn, d, m, _cur_mat(n, w, r))


def number_rate(dn: Any, bd: eu.Bands) -> float:
    """Return the weighted particle-number rate."""
    q = jnp.asarray(dn, dtype=F32)
    if q.shape != (2, bd.size):
        raise ValueError("dn must have shape (2, N)")
    w = jnp.asarray(bd.w, dtype=F32)
    return float(jax.device_get(jnp.sum(w * jnp.sum(q, axis=0))))


def check_db(r: Any, st: JState, t: float, eps: float = 1.0e-20) -> float:
    """Return the largest detailed-balance residual on the host."""
    rr = np.asarray(jax.device_get(r), dtype=np.float64)
    e = np.asarray(jax.device_get(st.e), dtype=np.float64).reshape(-1)
    rr = rr.reshape(e.size, e.size)
    de = e[:, None] - e[None, :]
    ma = (rr > eps) & (rr.T > eps)
    np.fill_diagonal(ma, False)
    if not np.any(ma):
        return float("nan")
    z = np.log(rr[ma]) - np.log(rr.T[ma]) + de[ma] / t
    return float(np.max(np.abs(z)))


def _fix_fill(n, w, pn):
    """Filling projection for Picard iteration, not for the time integrator."""
    nf = jnp.sum(w * jnp.sum(n, axis=0))
    df = pn - nf
    up = jnp.sum(w * jnp.sum(1.0 - n, axis=0))
    au = df / jnp.maximum(up, 1.0e-12)
    ad = df / jnp.maximum(nf, 1.0e-12)
    q = jnp.where(df >= 0.0, n + au * (1.0 - n), n + ad * n)
    return jnp.clip(q, 0.0, 1.0)


def _init_n(bd, p, n, d, m, tol, nmax, chk):
    """Shared checks; never silently repair an incorrect initial filling."""
    if chk <= 0 or nmax <= 0 or int(chk) != chk or int(nmax) != nmax:
        raise ValueError("chk and nmax must be positive integers")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not np.all(np.isfinite([d, m, p.v, p.h, p.n])):
        raise ValueError("mean-field parameters must be finite")
    if not 0 <= p.n <= 2:
        raise ValueError("n must be between 0 and 2")
    w = np.asarray(bd.w, dtype=np.float64)
    if np.any(~np.isfinite(w)) or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError("weights must be finite, nonnegative, and normalized")
    na = np.asarray(n, dtype=np.float64)
    if na.shape != (2, bd.size):
        raise ValueError("n must have shape (2, N)")
    if np.any(~np.isfinite(na)) or np.any(na < 0) or np.any(na > 1):
        raise ValueError("occupations must be finite and lie in the unit interval")
    nf = float(np.sum(w * np.sum(na, axis=0)))
    if abs(nf - p.n) > 5.0e-6:
        raise ValueError("initial occupations have the wrong filling")
    return jnp.asarray(na, dtype=F32)


def _result(bd, p, n, d, m, dn, er, it, tt, ok, hs):
    """Preserve the OpenSol/MFState API and the exact internal energy frame."""
    n0, dn0, d0, m0, t0, err = jax.device_get((n, dn, d, m, tt, er))
    n0 = np.asarray(n0, dtype=np.float64)
    dn0 = np.asarray(dn0, dtype=np.float64)
    d0, m0, t0, err = map(float, (d0, m0, t0, err))
    ep = np.finfo(np.float64).eps
    nc = np.clip(n0, ep, 1.0 - ep)
    eta = np.log1p(-nc) - np.log(nc)
    js = jax.device_get(mf_state(bd, p, d0, m0))
    # Do not use unshifted eu.mf_state energies in the returned solution.
    st0 = eu.MFState(
        bd, p, d0, m0, float(js.na), float(js.nb),
        np.asarray(js.eah), np.asarray(js.ebh), np.asarray(js.e),
        np.asarray(js.u), np.asarray(js.v),
    )
    hh = {q: np.asarray(z) for q, z in hs.items()}
    return eu.OpenSol(d0, m0, n0, eta, st0, dn0, err, it, t0, ok, hh)


def solve_open(
    bd: eu.Bands,
    p: eu.MFPars,
    n: Any,
    bs: Iterable[Bath],
    d: float = 0.2,
    m: float = 0.0,
    dt: float = 0.1,
    td: float = 0.5,
    tm: float = 0.5,
    tol: float = 1.0e-6,
    nmax: int = 50000,
    chk: int = 10,
    prog: bool = True,
    mode: str = "dense",
    block: int = 512,
) -> eu.OpenSol:
    """Integrate the instantaneous current with variable-step AB2.

    dt is a maximum step, NOT a guaranteed interval or an error tolerance.
    chk controls diagnostics/stopping only; it does not freeze the rates.
    hist['t'] and sol.time are accumulated accepted time.
    hist['h'] is the last accepted interval in each checkpoint block.
    AB2 history is preserved across checkpoints within this call.
    A new call starts a new trajectory with Euler startup.
    d and m retain the original phenomenological exponential relaxation.
    """
    n = _init_n(bd, p, n, d, m, tol, nmax, chk)
    if block <= 0 or int(block) != block:
        raise ValueError("block must be a positive integer")
    if mode not in ("dense", "block", "pair"):
        raise ValueError("mode must be 'dense', 'block', or 'pair'")
    if not np.all(np.isfinite([dt, td, tm])) or min(dt, td, tm) <= 0:
        raise ValueError("dt, td, and tm must be finite and positive")
    ea, eb, w, pv, pn, ph = _pars(bd, p)
    bp, rf = _baths(bs)
    if mode == "pair" and isinstance(bp, pb.Pack):
        raise ValueError("phonon baths require mode='dense' or mode='block'")
    d, m, tt = F32(d), F32(m), F32(0.0)
    dh, thd, thm = F32(dt), F32(td), F32(tm)
    dnp, hp, first = jnp.zeros_like(n), dh, jnp.asarray(True)
    hs = {q: [] for q in ("it", "t", "err", "cur", "ed", "em", "d", "m", "n0", "h")}
    ok, it, erp, tp = False, 0, None, 0.0
    bar = tqdm(range(0, int(nmax), int(chk)), desc="Open EI JAX", disable=not prog)
    for i0 in bar:
        ns = min(int(chk), int(nmax) - i0)
        n, d, m, tt, dnp, hp, first = _open_chunk(
            ea, eb, w, pv, pn, ph, bp, rf,
            n, d, m, tt, dnp, hp, first, dh, thd, thm,
            ns=ns, mode=mode, block=int(block),
        )
        dn, erj, ec, ed, em, n0 = _open_diag(
            ea, eb, w, pv, pn, ph, bp, rf, n, d, m, mode=mode, block=int(block)
        )
        er, ec0, ed0, em0, n00, d0, m0, t0, h0 = map(
            float, jax.device_get((erj, ec, ed, em, n0, d, m, tt, hp))
        )
        if not np.all(np.isfinite([er, ec0, ed0, em0, n00, d0, m0, t0, h0])):
            raise RuntimeError("nonfinite state or current during time evolution")
        if t0 <= tp or h0 <= 0:
            raise RuntimeError("time step collapsed; check rates and reduce dt")
        it = i0 + ns
        hs["it"].append(it)
        hs["t"].append(t0)
        hs["err"].append(er)
        hs["cur"].append(ec0)
        hs["ed"].append(ed0)
        hs["em"].append(em0)
        hs["d"].append(d0)
        hs["m"].append(m0)
        hs["n0"].append(n00)
        hs["h"].append(h0)
        if prog:
            bar.set_postfix(err=f"{er:.2e}", d=f"{d0:.5f}", m=f"{m0:.5f}")
        falling = erp is not None and er < erp
        # Permit an exactly stationary initial state as well.
        if er < tol and (falling or er == 0.0):
            ok = True
            break
        erp, tp = er, t0
    bar.close()
    return _result(bd, p, n, d, m, dn, erj, it, tt, ok, hs)


def solve_fixed(
    bd: eu.Bands,
    p: eu.MFPars,
    n: Any,
    bs: Iterable[Bath],
    d: float = 0.2,
    m: float = 0.0,
    mix: tuple[float, float, float] = (0.2, 0.2, 0.2),
    tol: float = 1.0e-6,
    nmax: int = 5000,
    chk: int = 10,
    prog: bool = True,
) -> eu.OpenSol:
    """Dense Picard iteration with a rate refresh every chk iterations.

    r is built once initially, frozen inside _fixed_chunk, and rebuilt at the
    checkpoint BEFORE evaluating the physical residual. The refreshed r is
    reused by the next chunk. Occupation-dependent a,b and mean-field targets
    are still updated on every iteration. Set chk=1 to refresh each iteration.
    hist['t'] and sol.time contain iteration counts for API compatibility;
    they must NOT be interpreted as physical time.
    """
    n = _init_n(bd, p, n, d, m, tol, nmax, chk)
    mx = np.asarray(mix, dtype=float)
    if mx.shape != (3,) or np.any(~np.isfinite(mx)) or np.any(mx <= 0) or np.any(mx > 1):
        raise ValueError("mix must contain three finite values in (0, 1]")
    ea, eb, w, pv, pn, ph = _pars(bd, p)
    bp, rf = _baths(bs)
    d, m = F32(d), F32(m)
    xn, xd, xm = map(F32, mx)
    st = _mf(ea, eb, pn, ph, d, m)
    r = _rates_jit(st, bp, rf)
    hs = {q: [] for q in ("it", "t", "err", "cur", "ed", "em", "d", "m", "n0", "gn", "na")}
    ok, it, erp = False, 0, None
    bar = tqdm(range(0, int(nmax), int(chk)), desc="Fixed EI JAX", disable=not prog)
    for i0 in bar:
        ns = min(int(chk), int(nmax) - i0)
        n, d, m = _fixed_chunk(ea, eb, w, pv, pn, ph, r, n, d, m, xn, xd, xm, ns=ns)
        st = _mf(ea, eb, pn, ph, d, m)
        r = _rates_jit(st, bp, rf)
        dn, erj, ec, ed, em, n0 = _fixed_diag(ea, eb, w, pv, pn, ph, r, n, d, m)
        tg = _targets(st, n, w, pv)
        gj = jnp.where(jnp.abs(d) > F32(1.0e-14), tg.d / d, jnp.nan)
        aj = jnp.sum(w * n[0])
        er, ec0, ed0, em0, n00, d0, m0, g0, a0 = map(
            float, jax.device_get((erj, ec, ed, em, n0, d, m, gj, aj))
        )
        if not np.all(np.isfinite([er, ec0, ed0, em0, n00, d0, m0, a0])):
            raise RuntimeError("nonfinite state or current during fixed iteration")
        it = i0 + ns
        hs["it"].append(it)
        hs["t"].append(float(it))
        hs["err"].append(er)
        hs["cur"].append(ec0)
        hs["ed"].append(ed0)
        hs["em"].append(em0)
        hs["d"].append(d0)
        hs["m"].append(m0)
        hs["n0"].append(n00)
        hs["gn"].append(g0)
        hs["na"].append(a0)
        if prog:
            bar.set_postfix(err=f"{er:.2e}", d=f"{d0:.5f}", m=f"{m0:.5f}")
        falling = erp is not None and er < erp
        if er < tol and (falling or er == 0.0):
            ok = True
            break
        erp = er
    bar.close()
    return _result(bd, p, n, d, m, dn, erj, it, float(it), ok, hs)


__all__ = [
    "Bath",
    "JState",
    "JTargets",
    "RateFn",
    "rate_db",
    "rate_product",
    "gam_db",
    "mf_state",
    "band_occ",
    "targets",
    "dense_rates",
    "dense_current",
    "blocked_current",
    "total_current",
    "number_rate",
    "check_db",
    "lim_step",
    "solve_open",
    "solve_fixed",
]
