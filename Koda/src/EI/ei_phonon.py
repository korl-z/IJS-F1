from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
from functools import lru_cache
from tqdm.auto import tqdm

import numpy as np
import jax
import jax.numpy as jnp

F = jnp.float64


@dataclass(frozen=True, eq=False)
class PhononBath:
    t: float
    k: Any
    cs: float
    eta: float
    gap: float
    ta: float
    tb: float
    dip: float
    amp: float = 1.0
    qd: float = np.pi
    a0: float = 1.0
    nang: int = 128
    name: str = "phonon"

    disp: str = "acoustic"
    w0: float = 1.0
    length: float = 1.0
    gam: float = 1.0e-3

    eh: float = 0.05
    em: float = 12.0

    def __post_init__(self):
        k = np.array(self.k, dtype=float, copy=True)
        z = np.asarray((
            self.t, self.cs, self.eta,
            self.amp, self.gap, self.ta, self.tb, self.dip, self.qd,
            self.a0, self.w0, self.length,
            self.gam, self.eh, self.em,
        ))

        if not np.isrealobj(z) or not np.isfinite(z).all():
            raise ValueError("bath parameters must be finite and real")

        if min(self.eta, self.qd, self.a0, self.eh, self.em) <= 0:
            raise ValueError("positive bath scales required")

        if min(self.cs, self.w0, self.gam, self.t, self.amp) < 0:
            raise ValueError("cs, w0, gam, t and amp must be nonnegative")

        if self.cs == 0.0 and self.w0 == 0.0:
            raise ValueError("cs and w0 cannot both be zero")
        
        qmax = np.sqrt(2.0) * np.pi / self.a0
        if self.qd > qmax * (1.0 + 1.0e-12):
            raise ValueError("qd must not exceed a corner of the first BZ")
        
        k.setflags(write=False)
        object.__setattr__(self, "k", k)


class Pack(NamedTuple):
    k: Any
    b: Any
    per: Any
    tab: Any
    flat: Any
    eg: Any
    eh: Any
    em: Any

# def omega_q(q, b):
#     q = jnp.asarray(q, dtype=F)

#     if b.disp == "constant":
#         return jnp.full_like(q, b.w0)

#     if b.disp == "acoustic":
#         return b.cs * q

#     wg = 2.0 * jnp.pi / b.length
#     return jnp.sqrt(wg * wg + (b.cs * q)**2)

def omega_q(q, b):
    q = jnp.asarray(q, dtype=F)
    return jnp.hypot(b.w0, b.cs * q)

@lru_cache(maxsize=8)
def _tab(nx, ny, na, eta, qd, per, nq):
    dx = (np.arange(nx) * per / nx)[:, None, None]
    dy = (np.arange(ny) * per / ny)[None, :, None]
    qr = np.linspace(0.0, qd, nq)[None, None, :]

    a = 2 * np.pi * eta / per
    r = np.exp(-a)
    c = -np.expm1(-2 * a) / per
    h = np.expm1(-a)**2
    qb = 0.5 * per

    def lor(x):
        return c / (
            h + 4 * r * np.sin(np.pi * x / per)**2
        )

    z = np.zeros((nx, ny, nq))
    th = (np.arange(na // 2) + 0.5) * (2 * np.pi / na)

    for v in tqdm(th, desc="Phonon kernel table", leave=False):
        qx = qr * np.cos(v)
        qy = qr * np.sin(v)

        ma = (np.abs(qx) <= qb) & (np.abs(qy) <= qb)

        z += ma * lor(dx - qx) * lor(dy - qy)
        z += ma * lor(dx + qx) * lor(dy + qy)

    z *= 2 * np.pi / na

    ix = (-np.arange(nx)) % nx
    iy = (-np.arange(ny)) % ny
    z = 0.5 * (z + z[ix][:, iy])

    z.setflags(write=False)
    return z


def pack(bs, nq=512):
    bs = tuple(bs)

    if not bs:
        raise ValueError("at least one phonon bath is required")
    if nq < 2:
        raise ValueError("nq must be at least 2")

    b = bs[0]

    if any(
        q.a0 != b.a0
        or q.nang != b.nang
        or q.eh != b.eh
        or q.em != b.em
        or not np.array_equal(q.k, b.k)
        for q in bs
    ):
        raise ValueError("baths must share k, a0 and nang")

    per = 2 * np.pi / b.a0
    kx, ky = np.unique(b.k[:, 0]), np.unique(b.k[:, 1])
    nx, ny = kx.size, ky.size

    if (
        nx * ny != len(b.k)
        or not np.allclose(np.diff(kx), per / nx)
        or not np.allclose(np.diff(ky), per / ny)
    ):
        raise ValueError(
            "cached kernels require a uniform Cartesian BZ grid"
        )


    pa = [
        (q.t, q.cs, q.eta, q.amp,
         q.ta, q.tb, q.dip, q.gap, q.qd, q.w0, q.gam)
        for q in bs
    ]

    tb = [_tab(nx, ny, int(q.nang), float(q.eta), float(q.qd), float(per), nq) for q in bs]

    fb = []

    ne = int(np.ceil(b.em / b.eh))
    eg = b.eh * np.arange(ne + 1, dtype=float)

    for q, z in zip(bs, tb):
        qr = np.linspace(0.0, q.qd, nq)

        if q.cs == 0.0:
            den = 1.0
            f = qr**3 / (q.w0 * den)
            v = np.trapezoid(z * f[None, None, :], qr, axis=-1)
        else:
            v = np.zeros((nx, ny), dtype=float)

        fb.append(v)

    return Pack(jnp.asarray(b.k, dtype=F), jnp.asarray(pa, dtype=F), F(per), jnp.asarray(np.stack(tb), dtype=F), jnp.asarray(np.stack(fb), dtype=F), jnp.asarray(eg, dtype=F), F(b.eh), F(eg[-1]))

def _wrap_q(q, per):
    return jnp.mod(q + 0.5 * per, per) - 0.5 * per

def g_mat(st, b, k, p, q):
    """Vertex M~[pol, nu, mu] of eqs. 125, 157 for final index k, source index p.

    q is the photon momentum k - p in the first BZ. pol 0 is TE (eq. 128),
    pol 1 is TM (eq. 129); nu is the final band, mu the source band.
    The rates use sum_pol |M~[pol, nu, mu]|^2.
    """
    q = jnp.asarray(q, dtype=F)
    kk = jnp.asarray(b.k, dtype=F)
    kf, ki = kk[k], kk[p]
    qm = jnp.linalg.norm(q)
    os = jnp.maximum(omega_q(qm, b), F(1.0e-30))
    g = jnp.where(qm <= b.qd, jnp.sqrt(b.amp / os), 0.0)

    ph = jnp.arctan2(q[1], q[0])
    cp, sn = jnp.cos(ph), jnp.sin(ph)
    ep = jnp.stack((jnp.stack((-sn, cp)), (b.w0 / os) * jnp.stack((cp, sn))))

    xv = ep @ (b.a0 * jnp.sin(b.a0 * (ki + 0.5 * q)))
    pv = b.dip * ep[:, 0]

    sf = jnp.sum(jnp.cos(b.a0 * kf))
    si = jnp.sum(jnp.cos(b.a0 * ki))
    eab = b.gap - 2.0 * b.ta * sf + 2.0 * b.tb * si
    eba = -b.gap - 2.0 * b.tb * sf + 2.0 * b.ta * si

    mm = jnp.stack((
        jnp.stack((2.0 * b.ta * xv, 1j * pv * eab), axis=-1),
        jnp.stack((1j * pv * eba, 2.0 * b.tb * xv), axis=-1),
    ), axis=-2)

    uk = jnp.array([
        [st.u[k], st.v[k]],
        [-st.v[k], st.u[k]],
    ])
    up = jnp.array([
        [st.u[p], st.v[p]],
        [-st.v[p], st.u[p]],
    ])

    return g * jnp.einsum("ba,lbc,cd->lad", uk, mm, up)


def _shell(dx, dy, q, i, bp):
    """Lookup the momentum difference and interpolate in phonon magnitude."""
    nx, ny, nq = bp.tab.shape[1:]
    ix = jnp.rint(dx * nx / bp.per).astype(jnp.int32) % nx
    iy = jnp.rint(dy * ny / bp.per).astype(jnp.int32) % ny

    z = jnp.clip(q * (nq - 1) / bp.b[i, 8], 0.0, nq - 1)
    iz = jnp.minimum(z.astype(jnp.int32), nq - 2)
    f = z - iz

    a = bp.tab[i, ix, iy, iz]
    b = bp.tab[i, ix, iy, iz + 1]
    return (1.0 - f) * a + f * b

def _flat_shell(dx, dy, i, bp):
    nx, ny = bp.flat.shape[1:]

    ix = jnp.rint(dx * nx / bp.per).astype(jnp.int32) % nx
    iy = jnp.rint(dy * ny / bp.per).astype(jnp.int32) % ny

    return bp.flat[i, ix, iy]


def _bose(w, t):
    x = w / jnp.where(t > 0.0, t, 1.0)
    n = 1.0 / jnp.maximum(jnp.expm1(x), F(1.0e-30))
    return jnp.where(t > 0.0, n, 0.0)


def _delta_e(x, gam):
    return gam / (jnp.pi * (x * x + gam * gam))


def _spec_e(w, w0, gam):
    d1 = (w - w0)**2 + gam**2
    d2 = (w + w0)**2 + gam**2

    sp = 8.0 * gam * w * w0 / (d1 * d2)

    return jnp.where(w0 > 0.0, sp, F(0.0))


def _spec_s(w0, gam):
    sl = 8.0 * gam * w0 / ((w0 * w0 + gam * gam)**2)

    return jnp.where(w0 > 0.0, sl, F(0.0))


def _hat(x, h):
    return jnp.maximum(F(1.0) - jnp.abs(x) / h, F(0.0)) / h


def _disc_e(w, w0, h):
    r = jnp.minimum(w0 / h, F(1.0))
    zn = jnp.where(w0 < h, r * (F(2.0) - r), F(1.0))
    zn = jnp.maximum(zn, F(1.0e-30))

    sp = F(2.0) * jnp.pi * (_hat(w - w0, h) - _hat(w + w0, h)) / zn

    return jnp.where(w0 > 0.0, jnp.maximum(sp, F(0.0)), F(0.0))


def _disc_s(w0, h):
    z = F(1.0e-6) * h
    return _disc_e(z, w0, h) / z


def _line(w, w0, gam, bp):
    ep = jnp.where(gam > 0.0, gam, bp.eh)

    sb = _spec_e(w, w0, ep)
    lb = _spec_s(w0, ep)

    sd = _disc_e(w, w0, bp.eh)
    ld = _disc_s(w0, bp.eh)

    sp = jnp.where(gam > 0.0, sb, sd)
    sl = jnp.where(gam > 0.0, lb, ld)

    ok = w <= bp.em
    sp = jnp.where(ok, sp, jnp.nan)
    sl = jnp.where(ok, sl, jnp.nan)

    return sp, sl


def _therm(sp, sl, w, t):
    ts = jnp.where(t > 0.0, t, F(1.0))
    x = w / ts

    dn = jnp.where(x > 0.0, -jnp.expm1(-x), F(1.0))

    ab0 = sp * jnp.exp(-x) / dn

    ab = jnp.where(t > 0.0, jnp.where(w > 0.0, ab0, sl * t), F(0.0))

    em = ab + sp
    return ab, em

def _bose2(w, t):
    """w**2 * N(w), including the exact zero-energy and T=0 limits."""
    x = w / jnp.where(t > 0, t, 1.0)
    den = jnp.where(x > 0, -jnp.expm1(-x), 1.0)
    wn = jnp.where(x > 0, w * jnp.exp(-x) / den, t)
    return jnp.where(t > 0, w * wn, 0.0)


def _data(st, bp):
    if bp.k.shape[0] != st.e.shape[1]:
        raise ValueError("bath k must follow the state momentum grid")
    a = jnp.concatenate((st.u, st.v))
    b = jnp.concatenate((-st.v, st.u))
    k = jnp.tile(bp.k, (2, 1))
    return jnp.column_stack((st.e.reshape(-1), a, b, k))


def _pair(x, y, bp):
    de = x[:, 0, None] - y[None, :, 0]
    w = jnp.abs(de)

    qx = _wrap_q(x[:, 3, None] - y[None, :, 3], bp.per)
    qy = _wrap_q(x[:, 4, None] - y[None, :, 4], bp.per)
    qm = jnp.hypot(qx, qy)

    aa = x[:, 1, None] * y[None, :, 1]
    ax = x[:, 1, None] * y[None, :, 2]
    xa = x[:, 2, None] * y[None, :, 1]
    bb = x[:, 2, None] * y[None, :, 2]

    # photon vertex geometry, eqs. 125, 128, 129 (dipole along x)
    a0 = 2.0 * jnp.pi / bp.per
    ph = jnp.arctan2(qy, qx)
    cp, sn = jnp.cos(ph), jnp.sin(ph)
    hx = a0 * jnp.sin(a0 * (y[None, :, 3] + 0.5 * qx))
    hy = a0 * jnp.sin(a0 * (y[None, :, 4] + 0.5 * qy))
    xt, xm = (-sn * hx + cp * hy)**2, (cp * hx + sn * hy)**2
    si = jnp.cos(a0 * x[:, 3, None]) + jnp.cos(a0 * x[:, 4, None])
    sj = jnp.cos(a0 * y[None, :, 3]) + jnp.cos(a0 * y[None, :, 4])

    def body(i, rr):
        (t, cs, eta, amp, ta, tb, dip, gap, qd, w0, gam) = bp.b[i]
        om = jnp.hypot(w0, cs * qm)
        os = jnp.maximum(om, F(1.0e-30))
        r2 = jnp.square(w0 / os)

        # bare band differences ea(k) - eb(k') and eb(k) - ea(k')
        eab = gap - 2.0 * ta * si + 2.0 * tb * sj
        eba = -gap - 2.0 * tb * si + 2.0 * ta * sj
        c1 = 2.0 * (ta * aa + tb * bb)
        c2 = dip * (eab * ax + eba * xa)
        af = ((xt + r2 * xm) * jnp.square(c1)
              + (sn * sn + r2 * cp * cp) * jnp.square(c2))

        g2 = amp * af / os

        ok = qm <= qd
        gm = jnp.where(ok, g2, F(0.0))

        sp, sl = _line(w, om, gam, bp)
        ab, em = _therm(sp, sl, w, t)

        r1 = gm * jnp.where(de < 0.0, em, ab)
        r2 = gm * jnp.where(de > 0.0, em, ab)

        return rr[0] + r1, rr[1] + r2

    z = jnp.zeros_like(de)

    return jax.lax.fori_loop( 0, bp.b.shape[0], body,(z, z),
    )


def rates(st, bp, j=None):
    """R[destination, source], with optional source columns j."""
    x = _data(st, bp)
    i = jnp.arange(x.shape[0])
    j = i if j is None else j
    r, _ = _pair(x, x[j], bp)
    return jnp.where(i[:, None] != j[None, :], r, 0.0)


@jax.jit
def _source(st, bp, j):
    return rates(st, bp, j).reshape(st.e.shape)


def source_rates(st, bs, mu, p):
    """All rates out of one source quasiparticle; shape (2, N)."""
    nk = st.e.shape[1]
    if mu not in (0, 1) or not 0 <= p < nk:
        raise ValueError("mu must be 0 or 1, and p must index the momentum grid")
    return _source(st, pack(bs), jnp.asarray([mu * nk + p]))


def cur_block(st, n, w, bp, block):
    """The existing Pauli current, evaluated in source blocks."""
    x = _data(st, bp)
    y, ww = n.reshape(-1), jnp.tile(w, 2)
    nm = y.size
    nb = (nm + block - 1) // block
    npad = nb * block - nm
    xp = jnp.pad(x, ((0, npad), (0, 0)))
    yp, wp = jnp.pad(y, (0, npad)), jnp.pad(ww, (0, npad))
    ii = jnp.arange(nm)[:, None]

    def body(ib, gl):
        j0 = ib * block
        xj = jax.lax.dynamic_slice_in_dim(xp, j0, block)
        yj = jax.lax.dynamic_slice_in_dim(yp, j0, block)
        wj = jax.lax.dynamic_slice_in_dim(wp, j0, block)
        jj = j0 + jnp.arange(block)
        ma = (jj[None, :] < nm) & (ii != jj[None, :])
        r1, r2 = _pair(x, xj, bp)
        r1, r2 = jnp.where(ma, r1, 0.0), jnp.where(ma, r2, 0.0)
        ga = gl[0] + r1 @ (wj * yj)
        lo = gl[1] + r2 @ (wj * (1.0 - yj))
        return ga, lo

    z = jnp.zeros_like(y)
    ga, lo = jax.lax.fori_loop(0, nb, body, (z, z))
    return ((1.0 - y) * ga - y * lo).reshape(n.shape)