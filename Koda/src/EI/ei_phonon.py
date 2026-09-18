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
    ktf: float
    eta: float
    amp: float = 1.0
    ca: float = 1.0
    cb: float = 1.0
    qd: float = np.pi
    a0: float = 1.0
    nang: int = 128
    name: str = "phonon"

    disp: str = "acoustic"
    w0: float = 1.0
    length: float = 1.0
    ew: float = 1.0e-3

    def __post_init__(self):
        k = np.array(self.k, dtype=float, copy=True)
        z = np.asarray((self.t, self.cs, self.ktf, self.eta,
                        self.amp, self.ca, self.cb, self.qd, self.a0))
        if k.ndim != 2 or k.shape[1] != 2 or not np.isfinite(k).all():
            raise ValueError("k must be a finite array with shape (N, 2)")
        if not np.isrealobj(z) or not np.isfinite(z).all():
            raise ValueError("bath parameters must be finite and real")
        if min(self.cs, self.ktf, self.eta, self.qd, self.a0) <= 0:
            raise ValueError("cs, ktf, eta, qd and a0 must be positive")
        if min(self.t, self.amp) < 0:
            raise ValueError("t and amp must be nonnegative")
        if self.nang < 4 or self.nang % 2 or int(self.nang) != self.nang:
            raise ValueError("nang must be an even integer of at least 4")
        
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


def omega_q(q, b):
    q = jnp.asarray(q, dtype=F)

    if b.disp == "constant":
        return jnp.full_like(q, b.w0)

    if b.disp == "acoustic":
        return b.cs * q

    wg = 2.0 * jnp.pi / b.length
    return jnp.sqrt(wg * wg + (b.cs * q)**2)

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

    dc = {
        "constant": 0,
        "acoustic": 1,
        "gapped": 2,
    }

    pa = [
        (
            q.t, q.cs, q.ktf, q.eta,
            q.amp, q.ca, q.cb, q.qd,
            dc[q.disp], q.w0, q.length, q.ew,
        )
        for q in bs
    ]

    tb = [
        _tab(
            nx, ny, int(q.nang), float(q.eta),
            float(q.qd), float(per), nq,
        )
        for q in bs
    ]

    fb = []

    for q, z in zip(bs, tb):
        qr = np.linspace(0.0, q.qd, nq)

        if q.disp == "constant":
            den = (qr * qr + q.ktf * q.ktf)**2
            f = qr**3 / (q.w0 * den)
            v = np.trapezoid(z * f[None, None, :], qr, axis=-1)
        else:
            v = np.zeros((nx, ny), dtype=float)

        fb.append(v)

    return Pack(jnp.asarray(b.k, dtype=F), jnp.asarray(pa, dtype=F), F(per), jnp.asarray(np.stack(tb), dtype=F), jnp.asarray(np.stack(fb), dtype=F))


def g_mat(st, b, k, p, q):
    """G[nu, mu] for one phonon momentum."""
    q = jnp.asarray(q, dtype=F)
    qm = jnp.linalg.norm(q)
    om = omega_q(qm, b)

    sg = jnp.where(q[0] != 0, jnp.sign(q[0]), jnp.sign(q[1]))

    den = qm**2 + b.ktf**2
    g = (-1j * sg * jnp.sqrt(b.amp) * qm / jnp.sqrt(jnp.maximum(om, F(1.0e-30))) / den)

    g = jnp.where((qm > 0.0) & (qm <= b.qd), g, 0.0)

    qb = jnp.pi / b.a0
    ma = ((jnp.abs(q[0]) <= qb) & (jnp.abs(q[1]) <= qb) & (qm <= b.qd))
    g = jnp.where(ma, g, 0.0)
    
    uk = jnp.array([
        [st.u[k], st.v[k]],
        [-st.v[k], st.u[k]],
    ])
    up = jnp.array([
        [st.u[p], st.v[p]],
        [-st.v[p], st.u[p]],
    ])

    c = jnp.diag(jnp.array([b.ca, b.cb]))
    return g * (uk.T @ c @ up)


def _shell(dx, dy, q, i, bp):
    """Lookup the momentum difference and interpolate in phonon magnitude."""
    nx, ny, nq = bp.tab.shape[1:]
    ix = jnp.rint(dx * nx / bp.per).astype(jnp.int32) % nx
    iy = jnp.rint(dy * ny / bp.per).astype(jnp.int32) % ny

    z = jnp.clip(q * (nq - 1) / bp.b[i, 7], 0.0, nq - 1)
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


def _delta_e(x, ew):
    return ew / (jnp.pi * (x * x + ew * ew))

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
    """Forward and reverse rates for all pairs of rows in x and y."""
    de = x[:, 0, None] - y[None, :, 0]
    w = jnp.abs(de)
    dx = x[:, 3, None] - y[None, :, 3]
    dy = x[:, 4, None] - y[None, :, 4]
    aa = x[:, 1, None] * y[None, :, 1]
    bb = x[:, 2, None] * y[None, :, 2]

    def body(i, rr):
        t, cs, ktf, eta, amp, ca, cb, qd, kind, w0, ll, ew = bp.b[i]

        af = (ca * aa + cb * bb)**2
        den0 = F(1.0e-30)

        # Acoustic and gapped branches.
        wg = 2.0 * jnp.pi / ll
        gap = jnp.where(kind == 2, wg, 0.0)

        q = jnp.sqrt(jnp.maximum(w * w - gap * gap, 0.0)) / cs
        den = (q * q + ktf * ktf)**2
        sh = _shell(dx, dy, q, i, bp)

        fa = amp * af * sh / (cs**4 * den)

        fg = (
            amp * af * sh * q * q
            / (
                cs * cs
                * jnp.maximum(w * w, den0)
                * den
            )
        )

        fd = jnp.where(kind == 2, fg, fa)
        vd = (kind != 0) & (w >= gap) & (q <= qd)
        fd = jnp.where(vd, fd, 0.0)

        bn = _bose2(w, t)

        d1 = fd * (
            bn + jnp.where(de < 0.0, w * w, 0.0)
        )
        d2 = fd * (
            bn + jnp.where(de > 0.0, w * w, 0.0)
        )

        # Constant branch.
        fc = amp * af * _flat_shell(dx, dy, i, bp)
        
        sp = _delta_e(w - w0, ew)
        ts = jnp.maximum(t, F(1.0e-12))
        nb = _bose(w0, t)
        z0 = F(1.0) + F(2.0) * nb
        # z0 = 1. #tail test

        c1t = fc * sp * z0 * jax.nn.sigmoid(-de / ts)
        c2t = fc * sp * z0 * jax.nn.sigmoid(de / ts)

        c10 = fc * sp * jnp.where(de < 0.0, F(1.0), F(0.0))
        c20 = fc * sp * jnp.where(de > 0.0, F(1.0), F(0.0))

        c1 = jnp.where(t > 0.0, c1t, c10)
        c2 = jnp.where(t > 0.0, c2t, c20)

        r1 = jnp.where(kind == 0, c1, d1)
        r2 = jnp.where(kind == 0, c2, d2)

        return rr[0] + r1, rr[1] + r2

    z = jnp.zeros_like(de)
    return jax.lax.fori_loop(0, bp.b.shape[0], body, (z, z))


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