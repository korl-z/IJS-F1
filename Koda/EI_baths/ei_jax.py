from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

import ei_unified as eu
# from Koda.EI_baths import ei_unified as eu

F32 = jnp.float32
I32 = jnp.int32
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


def rate_db(ei: Any, ej: Any, de: Any,
            t: Any, mu: Any) -> Any:
    """originalni rate za gamma_ij"""
    return jax.nn.sigmoid(-de / t)


def rate_product(ei: Any, ej: Any, de: Any,
                 t: Any, mu: Any) -> Any:
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
        a = np.array(
            [
                [kaa, kab],
                [kab, kbb],
            ],
            dtype=float,
        )

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
        if self.t <= 0.0 or self.wc <= 0.0:
            raise ValueError("t and wc must be positive, and kap nonnegative")
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
    return Bath(
        t=float(t),
        kap=kap,
        wc=float(wc),
        orb=orb,
        name=name,
        mu=float(mu),
        rate=rate,
    )


def _pars(bd: eu.Bands, p: eu.MFPars) -> tuple[Any, ...]:
    ea = jnp.asarray(bd.ea, dtype=F32)
    eb = jnp.asarray(bd.eb, dtype=F32)
    w = jnp.asarray(bd.w, dtype=F32)
    pv = jnp.asarray(p.v, dtype=F32)
    pn = jnp.asarray(p.n, dtype=F32)
    ph = jnp.asarray(p.h, dtype=F32)
    return ea, eb, w, pv, pn, ph


def _baths(bs: Iterable[Bath]) -> tuple[tuple[Any, ...], tuple[RateFn, ...]]:
    bs = tuple(bs)

    oc = {None: 0, "a": 1, "b": 2}
    bt = jnp.asarray([b.t for b in bs], dtype=F32)
    bk = jnp.asarray(np.stack([_kap_mat(b.kap) for b in bs]), dtype=F32)
    bw = jnp.asarray([b.wc for b in bs], dtype=F32)
    bo = jnp.asarray([oc[b.orb] for b in bs], dtype=I32)
    bm = jnp.asarray([b.mu for b in bs], dtype=F32)
    rf = tuple(b.rate for b in bs)
    return (bt, bk, bw, bo, bm), rf


def _mf(ea: Any, eb: Any, pn: Any, ph: Any,
        d: Any, m: Any) -> JState:
    na = F32(0.5) * (pn + m)
    nb = F32(0.5) * (pn - m)
    eah = ea + ph * nb
    ebh = eb + ph * na
    av = F32(0.5) * (eah + ebh)
    xi = F32(0.5) * (eah - ebh)
    ek = jnp.maximum(jnp.sqrt(xi * xi + d * d), F32(1.0e-14))
    z = jnp.clip(xi / ek, F32(-1.0), F32(1.0))
    u = jnp.sqrt(F32(0.5) * (F32(1.0) + z))
    v = jnp.copysign(jnp.sqrt(F32(0.5) * (F32(1.0) - z)), d)
    e = jnp.stack((av + ek, av - ek))
    return JState(d, m, na, nb, eah, ebh, e, u, v)


@jax.jit
def _mf_jit(ea: Any, eb: Any, pn: Any, ph: Any,
            d: Any, m: Any) -> JState:
    return _mf(ea, eb, pn, ph, d, m)


def mf_state(bd: eu.Bands, p: eu.MFPars,
             d: float, m: float) -> JState:
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


def targets(bd: eu.Bands, p: eu.MFPars,
            st: JState, n: Any) -> JTargets:
    """Return the device-side gap, imbalance, and filling targets."""
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


def _rates(st: JState, bp: tuple[Any, ...],
           rf: tuple[RateFn, ...]) -> Any:
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
def _rates_jit(st: JState, bp: tuple[Any, ...],
               rf: tuple[RateFn, ...]) -> Any:
    return _rates(st, bp, rf)


def dense_rates(st: JState, bs: Iterable[Bath]) -> Any:
    """Build the summed dense rate matrix on the JAX device."""
    bp, rf = _baths(bs)
    return _rates_jit(st, bp, rf)


def _cur_dense(st: JState, n: Any, w: Any,
               bp: tuple[Any, ...], rf: tuple[RateFn, ...]) -> Any:
    y = n.reshape(-1)
    ww = jnp.tile(w, 2)
    r = _rates(st, bp, rf)
    ga = (F32(1.0) - y) * (r @ (ww * y))
    lo = y * (r.T @ (ww * (F32(1.0) - y)))
    return (ga - lo).reshape(n.shape)


def _cur_block(st: JState, n: Any, w: Any, bp: tuple[Any, ...],
               rf: tuple[RateFn, ...], block: int) -> Any:
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
            qi = jnp.where(
                bo[ir] == 1,
                qa,
                jnp.where(bo[ir] == 2, qb, jnp.ones_like(qa)),
            )
            qj = jnp.where(
                bo[ir] == 1,
                aj,
                jnp.where(bo[ir] == 2, bj, jnp.ones_like(aj)),
            )
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
            lo = lo + jnp.sum(
                r2 * (wj * (F32(1.0) - yj))[None, :], axis=1
            )

        return ga, lo

    z = jnp.zeros_like(y)
    ga, lo = jax.lax.fori_loop(0, nb, src_body, (z, z))
    dn = (F32(1.0) - y) * ga - y * lo
    return dn.reshape(n.shape)

def _cur_pair(
    st: JState,
    n: Any,
    bp: tuple[Any, ...],
    rf: tuple[RateFn, ...],
) -> Any:
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

        af = jnp.where(
            bo[ir] == 0,
            jnp.ones_like(de),
            uv,
        )

        cf = bk[ir, 0, 1] * sp * af
        cr = bk[ir, 1, 0] * sp * af

        rab = rab + cf * fn(
            ea,
            eb,
            de,
            bt[ir],
            bm[ir],
        )

        rba = rba + cr * fn(
            eb,
            ea,
            -de,
            bt[ir],
            bm[ir],
        )

    q = (
        (F32(1.0) - na) * rab * nb
        - na * rba * (F32(1.0) - nb)
    )

    return jnp.stack((q, -q))


def _cur(st: JState, n: Any, w: Any, bp: tuple[Any, ...],
         rf: tuple[RateFn, ...], mode: str, block: int) -> Any:
    if mode == "pair":
        return _cur_pair(st, n, bp, rf)
    if mode == "dense":
        return _cur_dense(st, n, w, bp, rf)
    if mode == "block":
        return _cur_block(st, n, w, bp, rf, block)
    raise ValueError(
        "mode must be 'pair', 'dense', or 'block'"
    )


@partial(jax.jit, static_argnames=("rf", "mode", "block"))
def _cur_jit(st: JState, n: Any, w: Any, bp: tuple[Any, ...],
             rf: tuple[RateFn, ...], mode: str, block: int) -> Any:
    return _cur(st, n, w, bp, rf, mode, block)


def total_current(
    bd: eu.Bands,
    st: JState,
    n: Any,
    bs: Iterable[Bath],
    mode: str = "dense",
    block: int = 512,
) -> Any:
    """Evaluate the summed bath current on the JAX device."""
    if block <= 0:
        raise ValueError("block must be positive")
    n = jnp.asarray(n, dtype=F32)
    if n.shape != st.e.shape:
        raise ValueError("n must have shape (2, N)")
    w = jnp.asarray(bd.w, dtype=F32)
    bp, rf = _baths(bs)
    return _cur_jit(st, n, w, bp, rf, mode=mode, block=block)


def dense_current(bd: eu.Bands, st: JState,
                  n: Any, bs: Iterable[Bath]) -> Any:
    """Evaluate the current using a full rate matrix."""
    return total_current(bd, st, n, bs, mode="dense")


def blocked_current(
    bd: eu.Bands,
    st: JState,
    n: Any,
    bs: Iterable[Bath],
    block: int = 512,
) -> Any:
    """Evaluate the current using source blocks of the rate matrix."""
    return total_current(bd, st, n, bs, mode="block", block=block)


def _lim(n: Any, dn: Any, dt: Any, fac: Any = F32(0.8)) -> Any:
    jp = dn > 0.0
    jm = dn < 0.0
    dp = jnp.where(jp, dn, F32(1.0))
    dm = jnp.where(jm, dn, F32(-1.0))
    hp = jnp.min(jnp.where(jp, (F32(1.0) - n) / dp, jnp.inf))
    hm = jnp.min(jnp.where(jm, -n / dm, jnp.inf))
    hb = fac * jnp.maximum(F32(0.0), jnp.minimum(hp, hm))
    return jnp.minimum(dt, hb)


@jax.jit
def _lim_jit(n: Any, dn: Any, dt: Any, fac: Any) -> Any:
    return _lim(n, dn, dt, fac)


def lim_step(n: Any, dn: Any, dt: float, fac: float = 0.8) -> Any:
    """Return the device-side bounded Euler step."""
    return _lim_jit(
        jnp.asarray(n, dtype=F32),
        jnp.asarray(dn, dtype=F32),
        F32(dt),
        F32(fac),
    )


def _step(
    ca: tuple[Any, ...],
    _: Any,
    ea: Any,
    eb: Any,
    w: Any,
    pv: Any,
    pn: Any,
    ph: Any,
    bp: tuple[Any, ...],
    rf: tuple[RateFn, ...],
    dt: Any,
    td: Any,
    tm: Any,
    mode: str,
    block: int,
) -> tuple[tuple[Any, ...], None]:
    n, d, m, tt = ca
    st = _mf(ea, eb, pn, ph, d, m)
    dn = _cur(st, n, w, bp, rf, mode, block)
    h = _lim(n, dn, dt)
    n = jnp.clip(n + h * dn, 0.0, 1.0)
    n = _fix_fill(n, w, pn)
    tg = _targets(st, n, w, pv)
    zd = -jnp.expm1(-h / td)
    zm = -jnp.expm1(-h / tm)
    d = d + zd * (tg.d - d)
    m = m + zm * (tg.m - m)
    return (n, d, m, tt + h), None


@partial(jax.jit, static_argnames=("rf", "ns", "mode", "block"))
def _chunk(
    ea: Any,
    eb: Any,
    w: Any,
    pv: Any,
    pn: Any,
    ph: Any,
    bp: tuple[Any, ...],
    rf: tuple[RateFn, ...],
    n: Any,
    d: Any,
    m: Any,
    tt: Any,
    dt: Any,
    td: Any,
    tm: Any,
    ns: int,
    mode: str,
    block: int,
) -> tuple[Any, ...]:
    fn = partial(
        _step,
        ea=ea,
        eb=eb,
        w=w,
        pv=pv,
        pn=pn,
        ph=ph,
        bp=bp,
        rf=rf,
        dt=dt,
        td=td,
        tm=tm,
        mode=mode,
        block=block,
    )
    ca, _ = jax.lax.scan(fn, (n, d, m, tt), xs=None, length=ns)
    return ca


@partial(jax.jit, static_argnames=("rf", "mode", "block"))
def _diag(
    ea: Any,
    eb: Any,
    w: Any,
    pv: Any,
    pn: Any,
    ph: Any,
    bp: tuple[Any, ...],
    rf: tuple[RateFn, ...],
    n: Any,
    d: Any,
    m: Any,
    mode: str,
    block: int,
) -> tuple[Any, ...]:
    st = _mf(ea, eb, pn, ph, d, m)
    dn = _cur(st, n, w, bp, rf, mode, block)
    tg = _targets(st, n, w, pv)
    ec = jnp.max(jnp.abs(dn))
    ed = jnp.abs(tg.d - d)
    em = jnp.abs(tg.m - m)
    en = jnp.abs(tg.n - pn)
    er = jnp.maximum(jnp.maximum(ec, ed), jnp.maximum(em, en))
    return dn, er, ec, ed, em, tg.n


def number_rate(dn: Any, bd: eu.Bands) -> float:
    """Return the weighted particle-number rate."""
    q = jnp.asarray(dn, dtype=F32)
    if q.shape != (2, bd.size):
        raise ValueError("dn must have shape (2, N)")
    w = jnp.asarray(bd.w, dtype=F32)
    return float(jax.device_get(jnp.sum(w * jnp.sum(q, axis=0))))


def check_db(r: Any, st: JState, t: float,
             eps: float = 1.0e-20) -> float:
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
    nf = jnp.sum(w * jnp.sum(n, axis=0))
    df = pn - nf

    up = jnp.sum(w * jnp.sum(1.0 - n, axis=0))
    dn = jnp.sum(w * jnp.sum(n, axis=0))

    au = df / jnp.maximum(up, 1.0e-12)
    ad = df / jnp.maximum(dn, 1.0e-12)

    q = jnp.where(
        df >= 0.0,
        n + au * (1.0 - n),
        n + ad * n,
    )

    return jnp.clip(q, 0.0, 1.0)


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
    """Relax the open EI state in float32 on the selected JAX device."""
    if block <= 0 or chk <= 0 or nmax <= 0:
        raise ValueError("block, chk, and nmax must be positive")
    if dt <= 0.0 or td <= 0.0 or tm <= 0.0 or tol <= 0.0:
        raise ValueError("dt, td, tm, and tol must be positive")

    na = np.asarray(n, dtype=np.float32)
    if na.shape != (2, bd.size):
        raise ValueError("n must have shape (2, N)")
    if np.any(~np.isfinite(na)) or np.any(na < 0.0) or np.any(na > 1.0):
        raise ValueError("occupations must be finite and lie in the unit interval")
    nf = float(np.sum(bd.w * np.sum(na, axis=0)))
    if abs(nf - p.n) > 5.0e-6:
        raise ValueError("initial occupations have the wrong filling")

    ea, eb, w, pv, pn, ph = _pars(bd, p)
    bp, rf = _baths(bs)
    n = jnp.asarray(na, dtype=F32)
    d = jnp.asarray(d, dtype=F32)
    m = jnp.asarray(m, dtype=F32)
    tt = jnp.asarray(0.0, dtype=F32)
    dh = jnp.asarray(dt, dtype=F32)
    thd = jnp.asarray(td, dtype=F32)
    thm = jnp.asarray(tm, dtype=F32)

    hs = {q: [] for q in (
        "it", "t", "err", "cur", "ed", "em", "d", "m", "n0"
    )}
    ok = False
    it = 0
    er = np.inf
    tp = -np.inf
    bar = tqdm(range(0, nmax, chk), desc="Open EI JAX", disable=not prog)

    for i0 in bar:
        ns = min(chk, nmax - i0)
        n, d, m, tt = _chunk(
            ea,
            eb,
            w,
            pv,
            pn,
            ph,
            bp,
            rf,
            n,
            d,
            m,
            tt,
            dh,
            thd,
            thm,
            ns=ns,
            mode=mode,
            block=block,
        )
        dn, erj, ec, ed, em, n0 = _diag(
            ea,
            eb,
            w,
            pv,
            pn,
            ph,
            bp,
            rf,
            n,
            d,
            m,
            mode=mode,
            block=block,
        )
        er, ec0, ed0, em0, n00, d0, m0, t0 = map(
            float,
            jax.device_get((erj, ec, ed, em, n0, d, m, tt)),
        )
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
        if prog:
            bar.set_postfix(err=f"{er:.2e}", d=f"{d0:.5f}", m=f"{m0:.5f}")
        if er < tol:
            ok = True
            break
        if t0 <= tp:
            raise RuntimeError("occupation step collapsed to zero")
        tp = t0

    dn, erj, _, _, _, _ = _diag(
        ea,
        eb,
        w,
        pv,
        pn,
        ph,
        bp,
        rf,
        n,
        d,
        m,
        mode=mode,
        block=block,
    )
    n0, dn0, d0, m0, t0, er = jax.device_get((n, dn, d, m, tt, erj))
    n0 = np.asarray(n0, dtype=np.float32)
    dn0 = np.asarray(dn0, dtype=np.float32)
    d0 = float(d0)
    m0 = float(m0)
    t0 = float(t0)
    er = float(er)
    ep = np.finfo(np.float32).eps
    nc = np.clip(n0, ep, 1.0 - ep)
    eta = np.log((1.0 - nc) / nc)
    st0 = eu.mf_state(bd, p, d0, m0)
    hh = {q: np.asarray(z) for q, z in hs.items()}
    return eu.OpenSol(
        d0,
        m0,
        n0,
        eta,
        st0,
        dn0,
        er,
        it,
        t0,
        ok or er < tol,
        hh,
    )


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
]


# from __future__ import annotations

# from dataclasses import dataclass
# from functools import partial
# from typing import Any, Iterable, NamedTuple

# import numpy as np

# import jax
# import jax.numpy as jnp

# from tqdm.auto import tqdm

# import ei_unified as eu

# F32 = jnp.float32
# I32 = jnp.int32


# class JState(NamedTuple):
#     """Mean-field state stored on the JAX device."""

#     d: Any
#     m: Any
#     na: Any
#     nb: Any
#     eah: Any
#     ebh: Any
#     e: Any
#     u: Any
#     v: Any


# class JTargets(NamedTuple):
#     """Self-consistent targets stored on the JAX device."""

#     d: Any
#     m: Any
#     n: Any
#     na: Any
#     nb: Any


# def _kap_mat(kap):
#     """glede na aa, ab, bb scatteing"""
#     a = np.asarray(kap, dtype=float)

#     if a.ndim == 0:
#         z = float(a)
#         a = np.full((2, 2), z, dtype=float)

#     elif a.shape == (3,):
#         kaa, kab, kbb = map(float, a)
#         a = np.array(
#             [
#                 [kaa, kab],
#                 [kab, kbb],
#             ],
#             dtype=float,
#         )

#     return a


# @dataclass(frozen=True)
# class Bath:
#     """Parameters of a detailed-balance transfer bath."""

#     t: float
#     kap: float = 1.0
#     wc: float = np.inf
#     orb: str | None = None
#     name: str = "bath"
#     mu: float = 0.0

#     def __post_init__(self) -> None:
#         if self.t <= 0.0 or self.wc <= 0.0:
#             raise ValueError("t and wc must be positive, and kap nonnegative")
#         if self.orb not in (None, "a", "b"):
#             raise ValueError("orb must be None, 'a', or 'b'")

#         a = _kap_mat(self.kap)

#         q = tuple(tuple(float(x) for x in row) for row in a)

#         object.__setattr__(self, "kap", q)


# def gam_db(
#     t: float,
#     kap: Any = 1.0,
#     wc: float = np.inf,
#     orb: str | None = None,
#     name: str = "bath",
#     mu: float = 0.0,
# ) -> Bath:
#     """Return a JAX detailed-balance product bath."""
#     return Bath(
#         t=float(t),
#         kap=kap,
#         wc=float(wc),
#         orb=orb,
#         name=name,
#         mu=float(mu),
#     )


# def _pars(bd: eu.Bands, p: eu.MFPars) -> tuple[Any, ...]:
#     ea = jnp.asarray(bd.ea, dtype=F32)
#     eb = jnp.asarray(bd.eb, dtype=F32)
#     w = jnp.asarray(bd.w, dtype=F32)
#     pv = jnp.asarray(p.v, dtype=F32)
#     pn = jnp.asarray(p.n, dtype=F32)
#     ph = jnp.asarray(p.h, dtype=F32)
#     return ea, eb, w, pv, pn, ph


# def _baths(bs: Iterable[Bath]) -> tuple[Any, ...]:
#     bs = tuple(bs)

#     oc = {None: 0, "a": 1, "b": 2}
#     bt = jnp.asarray([b.t for b in bs], dtype=F32)
#     bk = jnp.asarray(np.stack([_kap_mat(b.kap) for b in bs]), dtype=F32)
#     bw = jnp.asarray([b.wc for b in bs], dtype=F32)
#     bo = jnp.asarray([oc[b.orb] for b in bs], dtype=I32)
#     bm = jnp.asarray([b.mu for b in bs], dtype=F32)
#     return bt, bk, bw, bo, bm


# def _mf(ea: Any, eb: Any, pn: Any, ph: Any, d: Any, m: Any) -> JState:
#     na = F32(0.5) * (pn + m)
#     nb = F32(0.5) * (pn - m)
#     eah = ea + ph * nb
#     ebh = eb + ph * na
#     av = F32(0.5) * (eah + ebh)
#     xi = F32(0.5) * (eah - ebh)
#     ek = jnp.maximum(jnp.sqrt(xi * xi + d * d), F32(1.0e-14))
#     z = jnp.clip(xi / ek, F32(-1.0), F32(1.0))
#     u = jnp.sqrt(F32(0.5) * (F32(1.0) + z))
#     v = jnp.copysign(jnp.sqrt(F32(0.5) * (F32(1.0) - z)), d)
#     e = jnp.stack((av + ek, av - ek))
#     return JState(d, m, na, nb, eah, ebh, e, u, v)


# @jax.jit
# def _mf_jit(ea: Any, eb: Any, pn: Any, ph: Any, d: Any, m: Any) -> JState:
#     return _mf(ea, eb, pn, ph, d, m)


# def mf_state(bd: eu.Bands, p: eu.MFPars, d: float, m: float) -> JState:
#     """Build a compatible mean-field state on the JAX device."""
#     ea, eb, _, _, pn, ph = _pars(bd, p)
#     return _mf_jit(ea, eb, pn, ph, F32(d), F32(m))


# def _band_occ(st: JState, n: Any) -> tuple[Any, Any]:
#     u2 = st.u * st.u
#     v2 = st.v * st.v
#     na = u2 * n[0] + v2 * n[1]
#     nb = v2 * n[0] + u2 * n[1]
#     return na, nb


# _band_occ_jit = jax.jit(_band_occ)


# def band_occ(st: JState, n: Any) -> tuple[Any, Any]:
#     """Rotate quasiparticle occupations back to the bare bands."""
#     n = jnp.asarray(n, dtype=F32)
#     if n.shape != st.e.shape:
#         raise ValueError("n must have shape (2, N)")
#     return _band_occ_jit(st, n)


# def _targets(st: JState, n: Any, w: Any, pv: Any) -> JTargets:
#     na, nb = _band_occ(st, n)
#     d = pv * jnp.sum(w * st.u * st.v * (n[1] - n[0]))
#     ma = jnp.sum(w * na)
#     mb = jnp.sum(w * nb)
#     nf = jnp.sum(w * jnp.sum(n, axis=0))
#     return JTargets(d, ma - mb, nf, ma, mb)


# @jax.jit
# def _targets_jit(st: JState, n: Any, w: Any, pv: Any) -> JTargets:
#     return _targets(st, n, w, pv)


# def targets(bd: eu.Bands, p: eu.MFPars, st: JState, n: Any) -> JTargets:
#     """Return the device-side gap, imbalance, and filling targets."""
#     n = jnp.asarray(n, dtype=F32)
#     if n.shape != st.e.shape:
#         raise ValueError("n must have shape (2, N)")
#     _, _, w, pv, _, _ = _pars(bd, p)
#     return _targets_jit(st, n, w, pv)


# def _qv(st: JState) -> tuple[Any, Any]:
#     qa = jnp.concatenate((st.u, st.v))
#     qb = jnp.concatenate((-st.v, st.u))
#     return qa, qb


# def _af(qa: Any, qb: Any, o: Any) -> Any:
#     q = jnp.where(o == 1, qa, jnp.where(o == 2, qb, jnp.ones_like(qa)))
#     a = q[:, None] * q[None, :]
#     return jnp.where(o == 0, jnp.ones_like(a), a * a)


# def _rates(st: JState, bp: tuple[Any, ...]) -> Any:
#     bt, bk, bw, bo, bm = bp

#     e = st.e.reshape(-1)
#     ei = e[:, None]
#     ej = e[None, :]
#     de = ei - ej

#     la = jnp.repeat(
#         jnp.arange(2, dtype=I32),
#         st.e.shape[1],
#     )

#     qa, qb = _qv(st)
#     r0 = jnp.zeros_like(de)

#     def body(i: int, r: Any) -> Any:
#         sp = jnp.exp(-jnp.square(de / bw[i]))

#         fi = jax.nn.sigmoid((bm[i] - ei) / bt[i])

#         hj = jax.nn.sigmoid((ej - bm[i]) / bt[i])

#         af = _af(qa, qb, bo[i])

#         km = bk[
#             i,
#             la[:, None],
#             la[None, :],
#         ]

#         return r + km * sp * fi * hj * af

#     r = jax.lax.fori_loop(
#         0,
#         bt.shape[0],
#         body,
#         r0,
#     )

#     ma = F32(1.0) - jnp.eye(e.size, dtype=F32)

#     return r * ma


# @jax.jit
# def _rates_jit(st: JState, bp: tuple[Any, ...]) -> Any:
#     return _rates(st, bp)


# def dense_rates(st: JState, bs: Iterable[Bath]) -> Any:
#     """Build the summed dense rate matrix on the JAX device."""
#     return _rates_jit(st, _baths(bs))


# def _cur_dense(st: JState, n: Any, w: Any, bp: tuple[Any, ...]) -> Any:
#     y = n.reshape(-1)
#     ww = jnp.tile(w, 2)
#     r = _rates(st, bp)
#     ga = (F32(1.0) - y) * (r @ (ww * y))
#     lo = y * (r.T @ (ww * (F32(1.0) - y)))
#     return (ga - lo).reshape(n.shape)


# def _cur_block(st: JState, n: Any, w: Any, bp: tuple[Any, ...], block: int) -> Any:
#     bt, bk, bw, bo, bm = bp
#     y = n.reshape(-1)
#     e = st.e.reshape(-1)
#     ww = jnp.tile(w, 2)
#     qa, qb = _qv(st)
#     nm = e.size
#     la = jnp.repeat(jnp.arange(2, dtype=I32), st.e.shape[1])
#     nb = (nm + block - 1) // block
#     npad = nb * block - nm
#     lp = jnp.pad(la, (0, npad))
#     ep = jnp.pad(e, (0, npad))
#     yp = jnp.pad(y, (0, npad))
#     wp = jnp.pad(ww, (0, npad))
#     ap = jnp.pad(qa, (0, npad))
#     bpv = jnp.pad(qb, (0, npad))
#     ii = jnp.arange(nm, dtype=I32)[:, None]

#     def src_body(ib: int, gl: tuple[Any, Any]) -> tuple[Any, Any]:
#         ga, lo = gl
#         j0 = ib * block
#         ej = jax.lax.dynamic_slice_in_dim(ep, j0, block)
#         yj = jax.lax.dynamic_slice_in_dim(yp, j0, block)
#         wj = jax.lax.dynamic_slice_in_dim(wp, j0, block)
#         aj = jax.lax.dynamic_slice_in_dim(ap, j0, block)
#         bj = jax.lax.dynamic_slice_in_dim(bpv, j0, block)
#         jj = j0 + jnp.arange(block, dtype=I32)
#         vm = jj < nm
#         pm = vm[None, :] & (ii != jj[None, :])
#         de = e[:, None] - ej[None, :]

#         lj = jax.lax.dynamic_slice_in_dim(lp, j0, block)
        
#         def bath_body(ir: int, xy: tuple[Any, Any]) -> tuple[Any, Any]:
#             xg, xl = xy
#             qi = jnp.where(
#                 bo[ir] == 1,
#                 qa,
#                 jnp.where(bo[ir] == 2, qb, jnp.ones_like(qa)),
#             )
#             qj = jnp.where(
#                 bo[ir] == 1,
#                 aj,
#                 jnp.where(bo[ir] == 2, bj, jnp.ones_like(aj)),
#             )
#             aa = qj[None, :] * qi[:, None]
#             af = jnp.where(bo[ir] == 0, jnp.ones_like(aa), aa * aa)
#             km = bk[
#                 ir,
#                 la[:, None],
#                 lj[None, :],
#             ]

#             sp = jnp.exp(
#                 -jnp.square(de / bw[ir])
#             )

#             c = km * sp * af * pm
#             # rf = c * jax.nn.sigmoid(-de / bt[ir])
#             # rr = c * jax.nn.sigmoid(de / bt[ir])

#             rf = (
#                 c
#                 * jax.nn.sigmoid((bm[ir] - e[:, None]) / bt[ir])
#                 * jax.nn.sigmoid((ej[None, :] - bm[ir]) / bt[ir])
#             )
#             rr = (
#                 c
#                 * jax.nn.sigmoid((bm[ir] - ej[None, :]) / bt[ir])
#                 * jax.nn.sigmoid((e[:, None] - bm[ir]) / bt[ir])
#             )

#             xg = xg + jnp.sum(rf * (wj * yj)[None, :], axis=1)
#             xl = xl + jnp.sum(rr * (wj * (F32(1.0) - yj))[None, :], axis=1)
#             return xg, xl

#         return jax.lax.fori_loop(0, bt.shape[0], bath_body, (ga, lo))

#     z = jnp.zeros_like(y)
#     ga, lo = jax.lax.fori_loop(0, nb, src_body, (z, z))
#     dn = (F32(1.0) - y) * ga - y * lo
#     return dn.reshape(n.shape)


# def _cur(st: JState, n: Any, w: Any, bp: tuple[Any, ...], mode: str, block: int) -> Any:
#     if mode == "dense":
#         return _cur_dense(st, n, w, bp)
#     if mode == "block":
#         return _cur_block(st, n, w, bp, block)
#     raise ValueError("mode must be 'dense' or 'block'")


# @partial(jax.jit, static_argnames=("mode", "block"))
# def _cur_jit(
#     st: JState, n: Any, w: Any, bp: tuple[Any, ...], mode: str, block: int
# ) -> Any:
#     return _cur(st, n, w, bp, mode, block)


# def total_current(
#     bd: eu.Bands,
#     st: JState,
#     n: Any,
#     bs: Iterable[Bath],
#     mode: str = "dense",
#     block: int = 512,
# ) -> Any:
#     """Evaluate the summed bath current on the JAX device."""
#     if block <= 0:
#         raise ValueError("block must be positive")
#     n = jnp.asarray(n, dtype=F32)
#     if n.shape != st.e.shape:
#         raise ValueError("n must have shape (2, N)")
#     w = jnp.asarray(bd.w, dtype=F32)
#     return _cur_jit(st, n, w, _baths(bs), mode=mode, block=block)


# def dense_current(bd: eu.Bands, st: JState, n: Any, bs: Iterable[Bath]) -> Any:
#     """Evaluate the current using a full rate matrix."""
#     return total_current(bd, st, n, bs, mode="dense")


# def blocked_current(
#     bd: eu.Bands, st: JState, n: Any, bs: Iterable[Bath], block: int = 512
# ) -> Any:
#     """Evaluate the current using source blocks of the rate matrix."""
#     return total_current(bd, st, n, bs, mode="block", block=block)


# def _lim(n: Any, dn: Any, dt: Any, fac: Any = F32(0.8)) -> Any:
#     jp = dn > 0.0
#     jm = dn < 0.0
#     dp = jnp.where(jp, dn, F32(1.0))
#     dm = jnp.where(jm, dn, F32(-1.0))
#     hp = jnp.min(jnp.where(jp, (F32(1.0) - n) / dp, jnp.inf))
#     hm = jnp.min(jnp.where(jm, -n / dm, jnp.inf))
#     hb = fac * jnp.maximum(F32(0.0), jnp.minimum(hp, hm))
#     return jnp.minimum(dt, hb)


# @jax.jit
# def _lim_jit(n: Any, dn: Any, dt: Any, fac: Any) -> Any:
#     return _lim(n, dn, dt, fac)


# def lim_step(n: Any, dn: Any, dt: float, fac: float = 0.8) -> Any:
#     """Return the device-side bounded Euler step."""
#     return _lim_jit(
#         jnp.asarray(n, dtype=F32),
#         jnp.asarray(dn, dtype=F32),
#         F32(dt),
#         F32(fac),
#     )


# def _step(
#     ca: tuple[Any, ...],
#     _: Any,
#     ea: Any,
#     eb: Any,
#     w: Any,
#     pv: Any,
#     pn: Any,
#     ph: Any,
#     bp: tuple[Any, ...],
#     dt: Any,
#     td: Any,
#     tm: Any,
#     mode: str,
#     block: int,
# ) -> tuple[tuple[Any, ...], None]:
#     n, d, m, tt = ca
#     st = _mf(ea, eb, pn, ph, d, m)
#     dn = _cur(st, n, w, bp, mode, block)
#     h = _lim(n, dn, dt)
#     n = jnp.clip(n + h * dn, 0.0, 1.0)
#     n = _fix_fill(n, w, pn)
#     tg = _targets(st, n, w, pv)
#     zd = -jnp.expm1(-h / td)
#     zm = -jnp.expm1(-h / tm)
#     d = d + zd * (tg.d - d)
#     m = m + zm * (tg.m - m)
#     return (n, d, m, tt + h), None


# @partial(jax.jit, static_argnames=("ns", "mode", "block"))
# def _chunk(
#     ea: Any,
#     eb: Any,
#     w: Any,
#     pv: Any,
#     pn: Any,
#     ph: Any,
#     bp: tuple[Any, ...],
#     n: Any,
#     d: Any,
#     m: Any,
#     tt: Any,
#     dt: Any,
#     td: Any,
#     tm: Any,
#     ns: int,
#     mode: str,
#     block: int,
# ) -> tuple[Any, ...]:
#     fn = partial(
#         _step,
#         ea=ea,
#         eb=eb,
#         w=w,
#         pv=pv,
#         pn=pn,
#         ph=ph,
#         bp=bp,
#         dt=dt,
#         td=td,
#         tm=tm,
#         mode=mode,
#         block=block,
#     )
#     ca, _ = jax.lax.scan(fn, (n, d, m, tt), xs=None, length=ns)
#     return ca


# @partial(jax.jit, static_argnames=("mode", "block"))
# def _diag(
#     ea: Any,
#     eb: Any,
#     w: Any,
#     pv: Any,
#     pn: Any,
#     ph: Any,
#     bp: tuple[Any, ...],
#     n: Any,
#     d: Any,
#     m: Any,
#     mode: str,
#     block: int,
# ) -> tuple[Any, ...]:
#     st = _mf(ea, eb, pn, ph, d, m)
#     dn = _cur(st, n, w, bp, mode, block)
#     tg = _targets(st, n, w, pv)
#     ec = jnp.max(jnp.abs(dn))
#     ed = jnp.abs(tg.d - d)
#     em = jnp.abs(tg.m - m)
#     en = jnp.abs(tg.n - pn)
#     er = jnp.maximum(jnp.maximum(ec, ed), jnp.maximum(em, en))
#     return dn, er, ec, ed, em, tg.n


# def number_rate(dn: Any, bd: eu.Bands) -> float:
#     """Return the weighted particle-number rate."""
#     q = jnp.asarray(dn, dtype=F32)
#     if q.shape != (2, bd.size):
#         raise ValueError("dn must have shape (2, N)")
#     w = jnp.asarray(bd.w, dtype=F32)
#     return float(jax.device_get(jnp.sum(w * jnp.sum(q, axis=0))))


# def check_db(r: Any, st: JState, t: float, eps: float = 1.0e-20) -> float:
#     """Return the largest detailed-balance residual on the host."""
#     rr = np.asarray(jax.device_get(r), dtype=np.float64)
#     e = np.asarray(jax.device_get(st.e), dtype=np.float64).reshape(-1)
#     rr = rr.reshape(e.size, e.size)
#     de = e[:, None] - e[None, :]
#     ma = (rr > eps) & (rr.T > eps)
#     np.fill_diagonal(ma, False)
#     if not np.any(ma):
#         return float("nan")
#     z = np.log(rr[ma]) - np.log(rr.T[ma]) + de[ma] / t
#     return float(np.max(np.abs(z)))


# def _fix_fill(n, w, pn):
#     nf = jnp.sum(w * jnp.sum(n, axis=0))
#     df = pn - nf

#     up = jnp.sum(w * jnp.sum(1.0 - n, axis=0))
#     dn = jnp.sum(w * jnp.sum(n, axis=0))

#     au = df / jnp.maximum(up, 1.0e-12)
#     ad = df / jnp.maximum(dn, 1.0e-12)

#     q = jnp.where(
#         df >= 0.0,
#         n + au * (1.0 - n),
#         n + ad * n,
#     )

#     return jnp.clip(q, 0.0, 1.0)


# def solve_open(
#     bd: eu.Bands,
#     p: eu.MFPars,
#     n: Any,
#     bs: Iterable[Bath],
#     d: float = 0.2,
#     m: float = 0.0,
#     dt: float = 0.1,
#     td: float = 0.5,
#     tm: float = 0.5,
#     tol: float = 1.0e-6,
#     nmax: int = 50000,
#     chk: int = 10,
#     prog: bool = True,
#     mode: str = "dense",
#     block: int = 512,
# ) -> eu.OpenSol:
#     """Relax the open EI state in float32 on the selected JAX device."""
#     if mode not in ("dense", "block"):
#         raise ValueError("mode must be 'dense' or 'block'")
#     if block <= 0 or chk <= 0 or nmax <= 0:
#         raise ValueError("block, chk, and nmax must be positive")
#     if dt <= 0.0 or td <= 0.0 or tm <= 0.0 or tol <= 0.0:
#         raise ValueError("dt, td, tm, and tol must be positive")

#     na = np.asarray(n, dtype=np.float32)
#     if na.shape != (2, bd.size):
#         raise ValueError("n must have shape (2, N)")
#     if np.any(~np.isfinite(na)) or np.any(na < 0.0) or np.any(na > 1.0):
#         raise ValueError("occupations must be finite and lie in the unit interval")
#     nf = float(np.sum(bd.w * np.sum(na, axis=0)))
#     if abs(nf - p.n) > 5.0e-6:
#         raise ValueError("initial occupations have the wrong filling")

#     ea, eb, w, pv, pn, ph = _pars(bd, p)
#     bp = _baths(bs)
#     n = jnp.asarray(na, dtype=F32)
#     d = jnp.asarray(d, dtype=F32)
#     m = jnp.asarray(m, dtype=F32)
#     tt = jnp.asarray(0.0, dtype=F32)
#     dh = jnp.asarray(dt, dtype=F32)
#     thd = jnp.asarray(td, dtype=F32)
#     thm = jnp.asarray(tm, dtype=F32)

#     hs = {q: [] for q in ("it", "t", "err", "cur", "ed", "em", "d", "m", "n0")}
#     ok = False
#     it = 0
#     er = np.inf
#     tp = -np.inf
#     bar = tqdm(range(0, nmax, chk), desc="Open EI JAX", disable=not prog)

#     for i0 in bar:
#         ns = min(chk, nmax - i0)
#         n, d, m, tt = _chunk(
#             ea,
#             eb,
#             w,
#             pv,
#             pn,
#             ph,
#             bp,
#             n,
#             d,
#             m,
#             tt,
#             dh,
#             thd,
#             thm,
#             ns=ns,
#             mode=mode,
#             block=block,
#         )
#         dn, erj, ec, ed, em, n0 = _diag(
#             ea,
#             eb,
#             w,
#             pv,
#             pn,
#             ph,
#             bp,
#             n,
#             d,
#             m,
#             mode=mode,
#             block=block,
#         )
#         er, ec0, ed0, em0, n00, d0, m0, t0 = map(
#             float,
#             jax.device_get((erj, ec, ed, em, n0, d, m, tt)),
#         )
#         it = i0 + ns
#         hs["it"].append(it)
#         hs["t"].append(t0)
#         hs["err"].append(er)
#         hs["cur"].append(ec0)
#         hs["ed"].append(ed0)
#         hs["em"].append(em0)
#         hs["d"].append(d0)
#         hs["m"].append(m0)
#         hs["n0"].append(n00)
#         if prog:
#             bar.set_postfix(err=f"{er:.2e}", d=f"{d0:.5f}", m=f"{m0:.5f}")
#         if er < tol:
#             ok = True
#             break
#         if t0 <= tp:
#             raise RuntimeError("occupation step collapsed to zero")
#         tp = t0

#     dn, erj, _, _, _, _ = _diag(
#         ea,
#         eb,
#         w,
#         pv,
#         pn,
#         ph,
#         bp,
#         n,
#         d,
#         m,
#         mode=mode,
#         block=block,
#     )
#     n0, dn0, d0, m0, t0, er = jax.device_get((n, dn, d, m, tt, erj))
#     n0 = np.asarray(n0, dtype=np.float32)
#     dn0 = np.asarray(dn0, dtype=np.float32)
#     d0 = float(d0)
#     m0 = float(m0)
#     t0 = float(t0)
#     er = float(er)
#     ep = np.finfo(np.float32).eps
#     nc = np.clip(n0, ep, 1.0 - ep)
#     eta = np.log((1.0 - nc) / nc)
#     st0 = eu.mf_state(bd, p, d0, m0)
#     hh = {q: np.asarray(z) for q, z in hs.items()}
#     return eu.OpenSol(
#         d0,
#         m0,
#         n0,
#         eta,
#         st0,
#         dn0,
#         er,
#         it,
#         t0,
#         ok or er < tol,
#         hh,
#     )


# __all__ = [
#     "Bath",
#     "JState",
#     "JTargets",
#     "gam_db",
#     "mf_state",
#     "band_occ",
#     "targets",
#     "dense_rates",
#     "dense_current",
#     "blocked_current",
#     "total_current",
#     "number_rate",
#     "check_db",
#     "lim_step",
#     "solve_open",
# ]
