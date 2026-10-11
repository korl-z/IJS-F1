"""Mean field state and rate matrices on the k grid, eqs. 182 and 186, 187.

g[i, j] is the rate from j = (band', k') to i = (band, k), eq. 186. The
reverse rate of eq. 187 is the transpose g[j, i]. This is the layout of the
dense rates in EI.ei_jax_2 (rows destination, columns source).
"""

from typing import Any, NamedTuple

import numpy as np

from .backend import host, jit, xp
from .bath import bose, klor, lor, pack
from .vertex import TINY, coh, pol, vert


class Prob(NamedTuple):
    """Device arrays of one problem: bands, weights, mean field and geometry."""

    ea: Any
    eb: Any
    w: Any
    pv: Any
    pn: Any
    ph: Any
    qm: Any
    x2: Any
    p2: Any
    ta: Any
    tb: Any


class State(NamedTuple):
    """Mean field state, same fields as ej.JState."""

    d: Any
    m: Any
    na: Any
    nb: Any
    eah: Any
    ebh: Any
    e: Any
    u: Any
    v: Any


def make_prob(bd, p, geo):
    """Collect eu.Bands, eu.MFPars and a Geom on the array backend."""
    if np.shape(geo.qm) != (bd.size, bd.size):
        raise ValueError("geo was built for a different k grid")

    def f(a):
        return xp.asarray(a, dtype=xp.float64)

    return Prob(f(bd.ea), f(bd.eb), f(bd.w), f(p.v), f(p.n), f(p.h),
                f(geo.qm), f(geo.x2), f(geo.p2), f(geo.ta), f(geo.tb))


def mf(pk, d, m):
    """Diagonalize the mean field Hamiltonian, eqs. 32, 33, 38, 44, 45.

    Same formulas and energy frame (shifted by -h/2) as ej._mf.
    """
    na = 0.5 * (pk.pn + m)
    nb = 0.5 * (pk.pn - m)
    sh = 0.5 * pk.ph
    eah = pk.ea + pk.ph * nb - sh
    ebh = pk.eb + pk.ph * na - sh
    av = 0.5 * (eah + ebh)
    xi = 0.5 * (eah - ebh)
    ek = xp.hypot(xi, d)
    sf = xp.where(ek > 0.0, ek, 1.0)
    lg = xp.sqrt((sf + xp.abs(xi)) / (2.0 * sf))
    sm = xp.abs(d) / (2.0 * sf * lg)
    u = xp.where(xi >= 0.0, lg, sm)
    v = xp.copysign(xp.where(xi >= 0.0, sm, lg), d)
    u = xp.where(ek > 0.0, u, 1.0)
    v = xp.where(ek > 0.0, v, 0.0)
    e = xp.stack((av + ek, av - ek))
    return State(d, m, na, nb, eah, ebh, e, u, v)


def tgt(st, n, pk):
    """Gap (eq. 37), imbalance (eq. 46) and filling from occupations n (2, N)."""
    d = pk.pv * xp.sum(pk.w * st.u * st.v * (n[1] - n[0]))
    m = xp.sum(pk.w * (st.u * st.u - st.v * st.v) * (n[0] - n[1]))
    nf = xp.sum(pk.w * (n[0] + n[1]))
    return d, m, nf


def rate_one(st, pk, b):
    """Rate matrix of one bath b = (t, w0, c, gam, amp), eq. 186."""
    t, w0, c, gam, amp = b
    s, om = vert(st, pk, w0, c, amp)
    e = st.e.reshape(-1)
    x = e[:, None] - e[None, :]
    g = s * klor(x, om, gam, t)
    return g * (1.0 - xp.eye(e.shape[0]))


def rate_mat(st, pk, bp):
    """Sum of the rate matrices of all baths in bp (rows of bath.pack)."""
    g = rate_one(st, pk, bp[0])
    for i in range(1, bp.shape[0]):
        g = g + rate_one(st, pk, bp[i])
    return g


def cur(n, w, g):
    """dn/dt of eq. 182 for occupations n (2, N) and a rate matrix g."""
    y = n.reshape(-1)
    ww = xp.concatenate((w, w))
    a = g @ (ww * y)
    b = g.T @ (ww * (1.0 - y))
    return ((1.0 - y) * a - y * b).reshape(n.shape)


@jit
def _rates(pk, bp, d, m):
    st = mf(pk, d, m)
    return st, rate_mat(st, pk, bp)


def dense_rates(bd, p, geo, bs, d, m):
    """Host copy of the summed rate matrix at (d, m), and the mean field state."""
    st, g = _rates(make_prob(bd, p, geo), pack(bs), d, m)
    return State(*(host(x) for x in st)), host(g)


def dissipator(bd, p, geo, bs, name="grid"):
    """The rates as an eu.Dissipator, for the time stepping eu.solve_open."""
    import EI.ei_unified as eu

    pk, bp, nk = make_prob(bd, p, geo), pack(bs), bd.size

    def gam(st, n):
        return host(rate_mat(st, pk, bp)).reshape(2, nk, 2, nk)

    return eu.Dissipator(gam, name=name)


def parts(bd, p, geo, b, d, m):
    """Separate factors of the rate matrix of one bath at (d, m), host arrays.

    (2N, 2N) arrays, [i, j] = (final, initial):
      x     Omega = E_i - E_j
      om    boson frequency om_q
      nb    Bose factor N(Omega), eq. 185
      ap    A_q(Omega), eq. 180;  am  A_q(-Omega), eq. 181
      nl    N(Omega) [A_q(Omega) - A_q(-Omega)], regular form
      m2te, m2tm   |M~|^2 of the TE and TM polarization
      m2v, m2d     velocity and dipole part of sum_pol |M~|^2
      m2    sum_pol |M~|^2 = m2te + m2tm = m2v + m2d
      g     rate matrix, eq. 186 (m2 nl with zero diagonal)
    (N, N) arrays: c1, c2 (4 each) from vertex.coh; xx, pp from vertex.pol.
    For w0 = 0 the q = 0 entries of m2 (1 / om_q) and nl (om_q) are not
    separately meaningful; g holds their finite product.
    """
    pk = make_prob(bd, p, geo)
    st = mf(pk, d, m)
    om, xx, pp = pol(pk, b.w0, b.c)
    c1, c2 = coh(st, pk)

    def blk(z):
        return xp.block([[z[0], z[1]], [z[2], z[3]]])

    go = xp.tile(b.amp / xp.maximum(om, TINY), (2, 2))
    r2 = xp.square(b.w0 / xp.maximum(om, TINY))
    te = go * blk([pk.x2[0] * c1[i] ** 2 + pk.p2[0] * c2[i] ** 2 for i in range(4)])
    tm = go * blk([r2 * (pk.x2[1] * c1[i] ** 2 + pk.p2[1] * c2[i] ** 2)
                   for i in range(4)])
    mv = go * blk([xx * c1[i] ** 2 for i in range(4)])
    md = go * blk([pp * c2[i] ** 2 for i in range(4)])
    e = st.e.reshape(-1)
    x = e[:, None] - e[None, :]
    o2 = xp.tile(om, (2, 2))
    out = dict(
        x=x, om=o2, nb=bose(x, b.t), ap=lor(x, o2, b.gam), am=lor(-x, o2, b.gam),
        nl=o2 * klor(x, o2, b.gam, b.t), m2te=te, m2tm=tm, m2v=mv, m2d=md,
        m2=te + tm, g=rate_one(st, pk, pack((b,))[0]),
        c1=xp.stack(c1), c2=xp.stack(c2), xx=xx, pp=pp,
        e=st.e, u=st.u, v=st.v,
    )
    return {q: host(z) for q, z in out.items()}
