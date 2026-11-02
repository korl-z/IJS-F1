from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

import EI.ei_unified as eu
import EI.ei_phonon as pb
import EI.ei_cavity as cv


class Ibz(NamedTuple):
    """C4v reduction of the tb_2d grid."""

    kf: Any  # full grid (N, 2), same order as eu.tb_2d
    ki: Any  # IBZ points (ni, 2), exact grid points
    idx: Any  # full point -> IBZ index, (N,)
    rep: Any  # IBZ index -> full point, (ni,)
    mult: Any  # orbit sizes m_j, (ni,)
    nk: int


class IbzBaths(NamedTuple):
    """Baths on the full grid plus the IBZ map, passed to the solvers."""

    bs: tuple
    ib: Ibz


class IbzPack(NamedTuple):
    pk: Any  # pb.Pack on the full grid
    ki: Any
    idx: Any
    rep: Any
    mult: Any


def ibz_grid(nk, a0=1.0):
    """Orbits of the tb_2d grid under C4v; nk must be even."""
    if nk % 2:
        raise ValueError("nk must be even")
    q = np.linspace(-np.pi / a0, np.pi / a0, nk, endpoint=False)
    kx, ky = np.meshgrid(q, q, indexing="ij")
    kf = np.stack((kx, ky), axis=-1).reshape(-1, 2)

    # integer coords around Gamma, mirrors fold to 0..h, swap folds hi >= lo
    h = nk // 2
    i, j = np.meshgrid(np.arange(nk), np.arange(nk), indexing="ij")
    x, y = np.abs(i - h), np.abs(j - h)
    key = (np.maximum(x, y) * (h + 1) + np.minimum(x, y)).reshape(-1)
    uk, idx, mult = np.unique(key, return_inverse=True, return_counts=True)

    # representative in the triangle 0 <= ky <= kx <= pi (pi stored as -pi)
    hi, lo = uk // (h + 1), uk % (h + 1)
    rep = ((hi + h) % nk) * nk + (lo + h) % nk
    return Ibz(kf, kf[rep], idx.reshape(-1), rep, mult.astype(float), nk)


def tb_2d_ibz(nk, gap, ta, tb, a0=1.0):
    """eu.tb_2d on the IBZ, with orbit weights m_j / N."""
    ib = ibz_grid(nk, a0)
    c = np.cos(ib.ki[:, 0] * a0) + np.cos(ib.ki[:, 1] * a0)
    ea = 0.5 * gap - 2.0 * ta * c
    eb = -0.5 * gap - 2.0 * tb * c
    bd = eu.Bands(ib.ki, ea, eb, ib.mult, (ib.ki.shape[0],))
    return bd, ib


def restrict(a, ib):
    """Full grid array (..., N) -> IBZ array (..., ni)."""
    return np.asarray(a)[..., ib.rep]


def unfold(a, ib):
    """IBZ array (..., ni) -> full grid (..., nk, nk)."""
    a = np.asarray(a)
    return a[..., ib.idx].reshape(a.shape[:-1] + (ib.nk, ib.nk))


def pack_ibz(ibs):
    pk = pb.pack(ibs.bs)
    ib = ibs.ib
    if not np.array_equal(np.asarray(pk.k), ib.kf):
        raise ValueError("bath k must be the full grid ib.kf")
    return IbzPack(pk, jnp.asarray(ib.ki), jnp.asarray(ib.idx),
                   jnp.asarray(ib.rep), jnp.asarray(ib.mult))


def rates_ibz(st, bp):
    """Orbit averaged R_bar[dest, src] on the IBZ.

    R_bar_ij = (1 / m_j) sum over k'' in orbit(j) of R(k_i, k''), with IBZ
    targets and full grid sources. Used with weights m_j / N it gives the
    exact full grid gain and loss.
    """
    ni, nf = bp.mult.size, bp.idx.size
    a = jnp.concatenate((st.u, st.v))
    b = jnp.concatenate((-st.v, st.u))
    e = st.e.reshape(-1)
    x = jnp.column_stack((e, a, b, jnp.tile(bp.ki, (2, 1))))

    # sources on the full grid, gathered from the IBZ state
    jf = jnp.concatenate((bp.idx, bp.idx + ni))
    y = jnp.column_stack((e[jf], a[jf], b[jf], jnp.tile(bp.pk.k, (2, 1))))
    r, _ = pb._pair(x, y, bp.pk)

    # drop only the identity term, same band and same full point
    rf = jnp.concatenate((bp.rep, bp.rep + nf))
    r = jnp.where(rf[:, None] != jnp.arange(2 * nf)[None, :], r, 0.0)

    # sum the source columns over each orbit, then average
    r = jax.ops.segment_sum(r.T, jf, num_segments=2 * ni).T
    r = r / jnp.tile(bp.mult, 2)[None, :]
    if bp.pk.cav is not None:
        r = r + cv.loc_cols(st.e, st.u, st.v, bp.ki, bp.mult / nf, bp.pk.cav,
                            jnp.arange(2 * ni))
    return r


# ---------------- s window on top of the IBZ ----------------


class WinBaths(NamedTuple):
    """IBZ baths with dynamics only for smin <= s <= smax, outside frozen."""

    bs: tuple
    ib: Ibz
    smin: float
    smax: float


class WinPack(NamedTuple):
    pk: Any
    ki: Any
    idx: Any
    rep: Any
    mult: Any
    wi: Any  # IBZ indices in the window
    wo: Any  # IBZ indices outside


class WinRates(NamedTuple):
    gw: Any  # (2 nw, 2 nw) window block of R_bar
    g: Any  # (2 nw,) gain from the frozen outside (n_beta = 1)
    l: Any  # (2 nw,) loss into the frozen outside (n_alpha = 0)
    wi: Any


def win_index(ib, smin, smax, a0=1.0):
    """IBZ indices inside and outside the s window."""
    s = np.cos(a0 * ib.ki[:, 0]) + np.cos(a0 * ib.ki[:, 1])
    m = (s >= smin) & (s <= smax)
    return np.flatnonzero(m), np.flatnonzero(~m)


def pack_win(wb):
    ip = pack_ibz(IbzBaths(wb.bs, wb.ib))
    wi, wo = win_index(wb.ib, wb.smin, wb.smax, wb.bs[0].a0)
    if wi.size == 0:
        raise ValueError("the s window holds no grid points")
    return WinPack(*ip, jnp.asarray(wi), jnp.asarray(wo))


def rates_win(st, bp):
    """Window block of R_bar plus the gain g and loss l from the frozen outside.

    Targets are the window states, sources all full grid states. The first
    output of _pair is the rate into the target, the second out of it.
    """
    ni, nf = bp.mult.size, bp.idx.size
    a = jnp.concatenate((st.u, st.v))
    b = jnp.concatenate((-st.v, st.u))
    e = st.e.reshape(-1)

    # window states as targets, full grid states as sources
    rw = jnp.concatenate((bp.wi, bp.wi + ni))
    x = jnp.column_stack((e[rw], a[rw], b[rw], jnp.tile(bp.ki[bp.wi], (2, 1))))
    jf = jnp.concatenate((bp.idx, bp.idx + ni))
    y = jnp.column_stack((e[jf], a[jf], b[jf], jnp.tile(bp.pk.k, (2, 1))))
    r1, r2 = pb._pair(x, y, bp.pk)

    # drop the identity term, then average the columns over each orbit
    rf = jnp.concatenate((bp.rep[bp.wi], bp.rep[bp.wi] + nf))
    keep = rf[:, None] != jnp.arange(2 * nf)[None, :]
    m2 = jnp.tile(bp.mult, 2)[None, :]

    def fold(r):
        r = jnp.where(keep, r, 0.0)
        return jax.ops.segment_sum(r.T, jf, num_segments=2 * ni).T / m2

    gr, lr = fold(r1), fold(r2)

    # frozen outside: n_alpha = 0, n_beta = 1, weights m_j / N
    ro = jnp.concatenate((bp.wo, bp.wo + ni))
    wo = jnp.tile(bp.mult[bp.wo], 2) / nf
    no = jnp.repeat(jnp.asarray([0.0, 1.0]), bp.wo.size)
    g = gr[:, ro] @ (wo * no)
    l = lr[:, ro] @ (wo * (1.0 - no))
    gw = gr[:, rw]
    if bp.pk.cav is not None:
        wi = bp.wi
        gw = gw + cv.loc_cols(st.e[:, wi], st.u[wi], st.v[wi], bp.ki[wi],
                              bp.mult[wi] / nf, bp.pk.cav,
                              jnp.arange(2 * wi.size))
    return WinRates(gw, g, l, bp.wi)


def ab_win(n, w, r):
    """Gain a and loss b on the IBZ state, zero outside the window."""
    y = n[:, r.wi].reshape(-1)
    ww = jnp.tile(w[r.wi], 2)
    a = r.gw @ (ww * y) + r.g
    b = r.gw.T @ (ww * (1.0 - y)) + r.l
    z = jnp.zeros_like(n)
    return z.at[:, r.wi].set(a.reshape(2, -1)), z.at[:, r.wi].set(b.reshape(2, -1))


def fix_fill_win(n, w, pn, r):
    """_fix_fill acting only on the window, the outside stays frozen."""
    ms = jnp.zeros(w.size, dtype=bool).at[r.wi].set(True)[None, :]
    df = pn - jnp.sum(w * jnp.sum(n, axis=0))
    up = jnp.sum(jnp.where(ms, w * (1.0 - n), 0.0))
    dw = jnp.sum(jnp.where(ms, w * n, 0.0))
    q = jnp.where(df >= 0.0, n + df / jnp.maximum(up, 1.0e-12) * (1.0 - n),
                  n + df / jnp.maximum(dw, 1.0e-12) * n)
    return jnp.where(ms, jnp.clip(q, 0.0, 1.0), n)


def freeze(n, bd, p, wb):
    """Initial IBZ state with the outside frozen and the filling fixed in the window."""
    wi, wo = win_index(wb.ib, wb.smin, wb.smax, wb.bs[0].a0)
    n = np.array(n, dtype=float)
    n[0, wo], n[1, wo] = 0.0, 1.0
    df = p.n - np.sum(bd.w * n.sum(axis=0))
    q = n[:, wi]
    if df >= 0.0:
        q = q + df / np.sum(bd.w[wi] * (1.0 - q)) * (1.0 - q)
    else:
        q = q + df / np.sum(bd.w[wi] * q) * q
    n[:, wi] = q
    return n
