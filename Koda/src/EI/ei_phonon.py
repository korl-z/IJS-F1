from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp

F = jnp.float64


@dataclass(frozen=True, eq=False)
class PhononBath:
    t: float
    k: Any
    cs: float
    gap: float
    ta: float
    tb: float
    dip: float
    amp: float = 1.0
    qd: float = 4.4428829381583662470158809900606936986146216893756902230853956069
    a0: float = 1.0
    name: str = "phonon"
    kind: str = "photon"  # photon: eq. 232, phonon: eq. 96 with D = 0
    ta_prime: float = 0.0  # Peierls slope t'_a, phonon only
    tb_prime: float = 0.0  # Peierls slope t'_b, phonon only

    w0: float = 1.0
    gam: float = 1.0e-3

    eh: float = 0.05  # hat line width, used only for gam = 0
    em: float | None = None  # largest |Omega|; None: bound from gap, ta, tb, v, n
    nsub: int = 16  # 0: line at the grid points, > 0: pair cell average

    v: float | None = None  # mean field V, only for the default em
    n: float = 1.0  # filling, only for the default em
    uh: float | None = None  # Hartree U if not V, only for the default em

    def __post_init__(self):
        k = np.array(self.k, dtype=float, copy=True)
        if self.em is None:
            if self.v is None:
                raise ValueError("give em, or v (and n, uh) for the default em")
            object.__setattr__(self, "em", em_bound(
                self.gap, self.ta, self.tb, self.v, self.n, self.uh))
        z = np.asarray((
            self.t, self.cs,
            self.amp, self.gap, self.ta, self.tb, self.dip, self.qd,
            self.a0, self.w0,
            self.gam, self.eh, self.em, self.ta_prime, self.tb_prime,
        ))

        if self.kind not in ("photon", "phonon"):
            raise ValueError("kind must be photon or phonon")
        if self.kind == "phonon" and self.dip != 0.0:
            raise ValueError("the phonon vertex has no dipole term, set dip = 0")

        if not np.isrealobj(z) or not np.isfinite(z).all():
            raise ValueError("bath parameters must be finite and real")

        if min(self.qd, self.a0, self.eh, self.em) <= 0:
            raise ValueError("positive bath scales required")

        if min(self.cs, self.w0, self.gam, self.t, self.amp) < 0:
            raise ValueError("cs, w0, gam, t and amp must be nonnegative")

        if self.cs == 0.0 and self.w0 == 0.0:
            raise ValueError("cs and w0 cannot both be zero")

        if self.nsub < 0 or int(self.nsub) != self.nsub:
            raise ValueError("nsub must be a nonnegative integer")
        if self.nsub > 0 and self.gam <= 0.0:
            raise ValueError("the cell average needs gam > 0")
        
        qmax = np.sqrt(2.0) * np.pi / self.a0
        if self.qd > qmax * (1.0 + 1.0e-12):
            raise ValueError("qd must not exceed a corner of the first BZ")
        
        k.setflags(write=False)
        object.__setattr__(self, "k", k)


def em_bound(gap, ta, tb, v, n=1.0, uh=None, mar=0.1):
    """max|E_i - E_j| forall d, m, plus margin just in case.

    E =  -(ta + tb) s +- sqrt(xi^2 + d^2) ,
    xi = G / 2 - (ta - tb) s - h m / 2, |s| <= 2, |m| <= min(n, 2 - n)
    |d| <= V / 2 (|u v| <= 1/2), h = V ce nimamo uh.
    """
    h = v if uh is None else uh
    xm = abs(gap) / 2 + 2 * abs(ta - tb) + abs(h) * min(n, 2 - n) / 2
    return (1 + mar) * (4 * abs(ta + tb) + 2 * np.hypot(xm, abs(v) / 2))


class Pack(NamedTuple):
    k: Any
    b: Any  # rows (t, cs, amp, ta, tb, dip, gap, qd, w0, gam, phn, pa, pb)
    per: Any
    eh: Any
    em: Any
    lt: Any = None  # cell averaged line tables (bath, 3, nx, ny, nw), or None
    dw: Any = None  # energy step of lt


def omega_q(q, b):
    q = jnp.asarray(q, dtype=F)
    return jnp.hypot(b.w0, b.cs * q)

@lru_cache(maxsize=8)
def _ltab(nx, ny, per, cs, w0, gam, qd, nsub, dw, nw, phn=False):
    """Pair cell average of P_c(q) [A(w) - A(-w)] / (w_q w), c = xx, yy, xy.

    For every grid transfer q_n the average runs over q_n + dq with the tent
    weight (1 - |dq_x| / dkx)(1 - |dq_y| / dky), which is the distribution of
    k - p for k and p uniform in their two cells. Pxx = sin^2 + r2 cos^2,
    Pyy = cos^2 + r2 sin^2, Pxy = 2 sin cos (r2 - 1), r2 = (w0 / w_q)^2.
    The line over w is regular at w = 0, so its value there is the slope.
    phn: phonon bath, r2 = 1, so P = (1, 1, 0) (L and T modes summed).
    """
    dkx, dky = per / nx, per / ny
    qx0 = np.mod(np.arange(nx) * dkx + 0.5 * per, per) - 0.5 * per
    qy0 = np.mod(np.arange(ny) * dky + 0.5 * per, per) - 0.5 * per
    t = 2.0 * ((np.arange(nsub) + 0.5) / nsub - 0.5)  # tent nodes in [-1, 1]
    wt = 1.0 - np.abs(t)
    wt = wt / wt.sum()
    w = dw * np.arange(nw)
    z = np.zeros((3, nx, ny, nw))
    for tx, ax in zip(t, wt):
        for ty, ay in zip(t, wt):
            qx = np.mod(qx0 + tx * dkx + 0.5 * per, per)[:, None] - 0.5 * per
            qy = np.mod(qy0 + ty * dky + 0.5 * per, per)[None, :] - 0.5 * per
            qm = np.hypot(qx, qy)
            ph = np.arctan2(qy, qx)
            cp, sn = np.cos(ph), np.sin(ph)
            om = np.hypot(w0, cs * qm)
            r2 = 1.0 if phn else (w0 / om) ** 2
            ok = (qm <= qd) * (ax * ay)
            pc = np.stack((sn * sn + r2 * cp * cp, cp * cp + r2 * sn * sn,
                           2.0 * sn * cp * (r2 - 1.0))) * ok
            om = om[..., None]
            ln = 8.0 * gam / (((w - om) ** 2 + gam**2) * ((w + om) ** 2 + gam**2))
            z += pc[..., None] * ln[None]
    z.setflags(write=False)
    return z


def pack(bs):
    bs = tuple(bs)

    if not bs:
        raise ValueError("at least one phonon bath is required")

    b = bs[0]

    if any(
        q.a0 != b.a0
        or q.eh != b.eh
        or q.em != b.em
        or q.nsub != b.nsub
        or not np.array_equal(q.k, b.k)
        for q in bs
    ):
        raise ValueError("baths must share k, a0, eh, em and nsub")

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
        (q.t, q.cs, q.amp,
         q.ta, q.tb, q.dip, q.gap, q.qd, q.w0, q.gam,
         float(q.kind == "phonon"), q.ta_prime, q.tb_prime)
        for q in bs
    ]

    lt, dw = None, None
    if b.nsub > 0:
        # energy step a fraction of the narrowest line, up to em
        dw = min(q.gam for q in bs) / 8.0
        nw = int(np.ceil(b.em / dw)) + 2
        lt = jnp.asarray(np.stack([
            _ltab(nx, ny, float(per), float(q.cs), float(q.w0), float(q.gam),
                  float(q.qd), int(q.nsub), float(dw), nw,
                  q.kind == "phonon") for q in bs]), dtype=F)
        dw = F(dw)

    return Pack(jnp.asarray(b.k, dtype=F), jnp.asarray(pa, dtype=F), F(per),
                F(b.eh), F(b.em), lt, dw)

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
    phn = b.kind == "phonon"  # pol 0 is T, pol 1 is L for phonons
    rt = 1.0 if phn else b.w0 / os
    ep = jnp.stack((jnp.stack((-sn, cp)), rt * jnp.stack((cp, sn))))

    if phn:
        xv = ep @ (jnp.sin(b.a0 * kf) - jnp.sin(b.a0 * ki))  # eq. 98
    else:
        xv = ep @ (b.a0 * jnp.sin(b.a0 * (ki + 0.5 * q)))
    pv = b.dip * ep[:, 0]
    ca, cb = (b.ta_prime, b.tb_prime) if phn else (b.ta, b.tb)

    sf = jnp.sum(jnp.cos(b.a0 * kf))
    si = jnp.sum(jnp.cos(b.a0 * ki))
    eab = b.gap - 2.0 * b.ta * sf + 2.0 * b.tb * si
    eba = -b.gap - 2.0 * b.tb * sf + 2.0 * b.ta * si

    mm = jnp.stack((
        jnp.stack((2.0 * ca * xv, 1j * pv * eab), axis=-1),
        jnp.stack((1j * pv * eba, 2.0 * cb * xv), axis=-1),
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


def _bose(w, t):
    x = w / jnp.where(t > 0.0, t, 1.0)
    n = 1.0 / jnp.maximum(jnp.expm1(x), F(1.0e-30))
    return jnp.where(t > 0.0, n, 0.0)


def _spec_e(w, w0, gam):
    d1 = (w - w0)**2 + gam**2
    d2 = (w + w0)**2 + gam**2

    return 8.0 * gam * w * w0 / (d1 * d2)


def _spec_s(w0, gam):
    return 8.0 * gam * w0 / ((w0 * w0 + gam * gam)**2)


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

    # fotonski vertex
    a0 = 2.0 * jnp.pi / bp.per
    ph = jnp.arctan2(qy, qx)
    cp, sn = jnp.cos(ph), jnp.sin(ph)
    hx = a0 * jnp.sin(a0 * (y[None, :, 3] + 0.5 * qx))
    hy = a0 * jnp.sin(a0 * (y[None, :, 4] + 0.5 * qy))
    xt, xm = (-sn * hx + cp * hy)**2, (cp * hx + sn * hy)**2
    si = jnp.cos(a0 * x[:, 3, None]) + jnp.cos(a0 * x[:, 4, None])
    sj = jnp.cos(a0 * y[None, :, 3]) + jnp.cos(a0 * y[None, :, 4])

    # phonon vertex, eq. 98: sin k - sin k', x final, y source
    gx = jnp.sin(a0 * x[:, 3, None]) - jnp.sin(a0 * y[None, :, 3])
    gy = jnp.sin(a0 * x[:, 4, None]) - jnp.sin(a0 * y[None, :, 4])

    if bp.lt is not None:
        # cell averaged line: grid index of q and linear interpolation in w
        nx, ny, nw = bp.lt.shape[2:]
        ix = jnp.rint(qx * nx / bp.per).astype(jnp.int32) % nx
        iy = jnp.rint(qy * ny / bp.per).astype(jnp.int32) % ny
        z = jnp.clip(w / bp.dw, 0.0, nw - 1.0)
        iw = jnp.minimum(z.astype(jnp.int32), nw - 2)
        fw = z - iw

    def body(i, rr):
        (t, cs, amp, ta, tb, dip, gap, qd, w0, gam, phn, ta_prime, tb_prime) = bp.b[i]
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

        # phonon, eq. 96 with D = 0: t -> t', h -> sin k - sin k', Z -> 1
        c1 = jnp.where(phn > 0, 2.0 * (ta_prime * aa + tb_prime * bb), c1)
        af = jnp.where(phn > 0, (gx * gx + gy * gy) * jnp.square(c1), af)

        g2 = amp * af / os

        ok = qm <= qd
        gm = jnp.where(ok, g2, F(0.0))

        if bp.lt is None:
            sp, sl = _line(w, om, gam, bp)
        else:
            # vertex with P(q) and the line averaged over the cell pair
            tt = bp.lt[i]
            tv = (1.0 - fw) * tt[:, ix, iy, iw] + fw * tt[:, ix, iy, iw + 1]
            ux = jnp.where(phn > 0, gx, hx)
            uy = jnp.where(phn > 0, gy, hy)
            sl = amp * ((ux * ux * tv[0] + uy * uy * tv[1] + ux * uy * tv[2])
                        * jnp.square(c1) + tv[0] * jnp.square(c2))
            sl = jnp.where(w <= bp.em, sl, jnp.nan)
            sp, gm = sl * w, F(1.0)
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