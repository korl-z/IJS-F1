"""Rate matrix R[dest, src] on the kinetic states (alpha, beta) x g.wi. JAX, float64.

Same equations and layout as cpu/rates.py. R is the C4v average over the images of the
source, so the solvers use a = R (w n), b = R^T (w (1 - n)), dn = (1 - n) a - n b.
mf and targets are the jax copies of model.mf and model.targets (needed under jit).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import expit

from ..model import State


_TAB = {}


def dev(x):
    """Device copy of a Grid or Pack. The large line tables are uploaded once and reused."""
    def up(a):
        if not isinstance(a, np.ndarray) or a.size < 65536:
            return jnp.asarray(a)
        if id(a) not in _TAB:
            if len(_TAB) >= 16:
                _TAB.clear()
            _TAB[id(a)] = (a, jnp.asarray(a))  # holding a keeps its id valid
        return _TAB[id(a)][1]

    return jax.tree_util.tree_map(up, x)


def mf(md, g, d, m):
    """Bogoliubov state (eq. 33, 44, 45), c_a = u alpha + v beta, c_b = -v alpha + u beta."""
    na, nb = 0.5 * (md.n + m), 0.5 * (md.n - m)
    eah = 0.5 * md.gap - 2.0 * md.ta * g.s + md.h * (nb - 0.5)
    ebh = -0.5 * md.gap - 2.0 * md.tb * g.s + md.h * (na - 0.5)
    xi, av = 0.5 * (eah - ebh), 0.5 * (eah + ebh)
    ek = jnp.hypot(xi, d)
    sf = jnp.where(ek > 0.0, ek, 1.0)
    big = jnp.sqrt((sf + jnp.abs(xi)) / (2.0 * sf))
    sml = jnp.abs(d) / (2.0 * sf * big)
    u = jnp.where(ek > 0.0, jnp.where(xi >= 0.0, big, sml), 1.0)
    v = jnp.where(ek > 0.0, jnp.copysign(jnp.where(xi >= 0.0, sml, big), d), 0.0)
    return State(jnp.stack((av + ek, av - ek)), u, v)


def targets(md, g, st, n):
    """Gap, imbalance and filling from the occupations (eq. 37, 42, 43)."""
    u2, v2 = st.u**2, st.v**2
    na = jnp.sum(g.w * (u2 * n[0] + v2 * n[1]))
    nb = jnp.sum(g.w * (v2 * n[0] + u2 * n[1]))
    d = md.v * jnp.sum(g.w * st.u * st.v * (n[1] - n[0]))
    return d, na - nb, na + nb


def therm(de, t):
    """Omega N(Omega) for Omega = E_dest - E_src: absorption w N(w), emission w (1 + N(w))."""
    w = jnp.abs(de)
    x = w / jnp.where(t > 0.0, t, 1.0)
    bw = jnp.where(x > 1.0e-8, w / jnp.expm1(jnp.clip(x, 1.0e-8, 700.0)), t)
    return jnp.where(t > 0.0, bw, 0.0) + jnp.maximum(-de, 0.0)


def kern(om, w0, gam):
    """Bracket of eq. 286: 4 gam Om int_w0^inf (w + w0^2 / w) / (d1 d2) dw."""
    z4 = (om**2 + gam**2) ** 2
    a = jnp.arctan((om - w0) / gam) + jnp.arctan((om + w0) / gam)
    l = jnp.log(((om - w0) ** 2 + gam**2) * ((om + w0) ** 2 + gam**2) / w0**4)
    return (1.0 + w0**2 * (om**2 - gam**2) / z4) * a + gam * om * w0**2 / z4 * l


@jax.jit
def _toy(p, e, qa, qb, bd):
    """Sum of the toy baths, rows p = (t, wc, orb, mu, prod, kaa, kab, kba, kbb)."""
    de = e[:, None] - e[None, :]
    bi, bj = bd[:, None], bd[None, :]
    r = jnp.zeros_like(de)
    for t, wc, orb, mu, prod, kaa, kab, kba, kbb in p:
        kap = jnp.where(bi == 0, jnp.where(bj == 0, kaa, kab), jnp.where(bj == 0, kba, kbb))
        q = jnp.where(orb == 1.0, qa, jnp.where(orb == 2.0, qb, 1.0))
        f = jnp.where(prod > 0.0, expit((mu - e) / t)[:, None] * expit((e - mu) / t)[None, :],
                     expit(-de / t))
        r = r + kap * jnp.exp(-((de / wc) ** 2)) * f * (q[:, None] * q[None, :]) ** 2
    return r


def _look(tab, cen, iq, w, gam, hh):
    """Line tables at energy w and transfer iq, linear in the sinh grid index."""
    mm = (tab.shape[-1] - 1) // 2
    z = jnp.clip(jnp.arcsinh((w - cen[:, iq]) / gam) / hh + mm, 0.0, 2.0 * mm)
    i0 = jnp.minimum(z.astype(int), 2 * mm - 1)
    f = z - i0
    c = jnp.arange(tab.shape[0]).reshape(-1, *([1] * (i0.ndim - 1)))
    return (1.0 - f) * tab[c, iq, i0] + f * tab[c, iq, i0 + 1]


@jax.jit
def _lin(md, g, b, hh, e, qa, qb, s, ii):
    """Rows ii of one grid bath: amp [C^2 <vertex Z F> + D^2 <F_TM2>] Omega N(Omega)."""
    par, tab, cen = b
    t, amp, gam, c1a, c1b, dip = par
    nw, nr = g.wi.shape[0], ii.shape[0]
    kt = g.ki[g.wi][ii % nw][:, None, None, :]  # targets (nr, 1, 1, 2)
    ks = g.kg[g.wi][None]  # sources and images (1, nw, 8, 2)
    qi = (kt - ks) % g.nk
    iq = (qi[..., 0] * g.nk + qi[..., 1])[:, None]  # (nr, 1, nw, 8)
    de = e[ii][:, None] - e[None, :]
    tv = _look(tab, cen, iq, jnp.abs(de).reshape(nr, 2, nw, 1), gam, hh)  # (rows, nr, 2, nw, 8)
    aa, bb = qa[ii][:, None] * qa[None, :], qb[ii][:, None] * qb[None, :]
    if tab.shape[0] == 1:  # phonon: |sin k - sin k'|^2, L and T summed
        v2 = jnp.sum((g.sn[kt] - g.sn[ks]) ** 2, -1)[:, None]
        sm = (2.0 * (c1a * aa + c1b * bb)) ** 2 * jnp.mean(v2 * tv[0], -1).reshape(nr, -1)
    else:  # photon: h = sin(k' + q / 2), q in the first BZ
        qc = (qi + g.nk // 2) % g.nk - g.nk // 2
        h = g.s2[(2 * ks + qc) % (2 * g.nk)]
        hx, hy = h[..., 0][:, None], h[..., 1][:, None]
        x1 = jnp.mean(hx * hx * tv[0] + hy * hy * tv[1] + hx * hy * tv[2], -1).reshape(nr, -1)
        si, sj = s[ii][:, None], s[None, :]
        eab = md.gap - 2.0 * md.ta * si + 2.0 * md.tb * sj  # eq. 233
        eba = -md.gap - 2.0 * md.tb * si + 2.0 * md.ta * sj
        c2 = dip * (eab * qa[ii][:, None] * qb[None, :] + eba * qb[ii][:, None] * qa[None, :])
        sm = (2.0 * (c1a * aa + c1b * bb)) ** 2 * x1 + c2**2 * jnp.mean(tv[3], -1).reshape(nr, -1)
    return amp * sm * therm(de, t)


@jax.jit
def _cho(g, b, e, qa, qb):
    """One chord bath: R = amp C^2 Omega N(Omega) L(I, J, |Omega|) / (w_I w_J).

    L = 1 / (2 pi) int q dq F(Omega, omega_q) S(I, J, q), F the Lorentzian of eq. 272.
    gam = 0: F -> 2 pi delta(|Omega| - cs q) / (cs q)^2, so L = S(q*) / (cs^2 |Omega|),
    q* = |Omega| / cs, zero beyond the Debye radius.
    """
    par, S, dq = b
    t, amp, gam, tpa, tpb, cs, w0, qd = par
    nw, nq = g.wi.shape[0], S.shape[-1]
    q = dq * jnp.arange(nq)
    om = jnp.hypot(w0, cs * q)
    wq = jnp.where(q <= qd, q * dq / (2.0 * jnp.pi), 0.0)
    gs = jnp.where(gam > 0.0, gam, 1.0)
    de = e[:, None] - e[None, :]
    i, j = jnp.arange(nw)[:, None], jnp.arange(nw)[None, :]
    blk = []
    for a in (0, 1):  # band blocks, all share the table
        row = []
        for c in (0, 1):
            w = jnp.abs(de[a * nw:(a + 1) * nw, c * nw:(c + 1) * nw])
            f = 8.0 * gs / (((w[..., None] - om) ** 2 + gs**2) * ((w[..., None] + om) ** 2 + gs**2))
            z = jnp.minimum(w / (cs * dq), nq - 1.0)
            l0 = jnp.minimum(z.astype(int), nq - 2)
            sh = ((1.0 - (z - l0)) * S[i, j, l0] + (z - l0) * S[i, j, l0 + 1]) / (cs**2 * jnp.maximum(w, 1.0e-300))
            ok = (w > 0.0) & (w <= cs * jnp.minimum(qd, (nq - 1) * dq))
            row.append(jnp.where(gam > 0.0, jnp.sum(S * f * wq, -1), jnp.where(ok, sh, 0.0)))
        blk.append(row)
    ww = jnp.concatenate((g.w[g.wi], g.w[g.wi]))
    c1 = 2.0 * (tpa * qa[:, None] * qa[None, :] + tpb * qb[:, None] * qb[None, :])
    return amp * c1**2 * jnp.block(blk) * therm(de, t) / (ww[:, None] * ww[None, :])


@jax.jit
def _cav(md, g, st, p):
    """Vertical rates up (beta_k -> alpha_k) and dn (alpha_k -> beta_k), eq. 286, 287."""
    om = (st.e[0] - st.e[1])[g.wi]
    cc = g.hv[g.wi] * (2.0 * (md.ta - md.tb) * st.u[g.wi] * st.v[g.wi]) ** 2
    up, dn = jnp.zeros_like(om), jnp.zeros_like(om)
    for t, w0, gam, amp, c in p:
        j = amp / (2.0 * jnp.pi * c**2) * cc * kern(om, w0, gam)
        nb = jnp.where(t > 0.0, 1.0 / jnp.maximum(jnp.expm1(om / jnp.where(t > 0.0, t, 1.0)), 1.0e-300), 0.0)
        up, dn = up + j * nb, dn + j * (1.0 + nb)
    return up, dn


def rates(md, g, st, pk, rows=None):
    """R[dest, src] of all baths at the state st. rows: target rows per chunk (memory).

    Pass g and pk through dev() once; numpy inputs work but are copied on every call.
    """
    wi = g.wi
    nw = wi.shape[0]
    e = st.e[:, wi].reshape(-1)
    qa = jnp.concatenate((st.u[wi], st.v[wi]))  # a orbital weight of (alpha, beta)
    qb = jnp.concatenate((-st.v[wi], st.u[wi]))
    s = jnp.concatenate((g.s[wi], g.s[wi]))
    r = jnp.zeros((2 * nw, 2 * nw))
    if pk.toy is not None:
        r = r + _toy(pk.toy, e, qa, qb, jnp.arange(2 * nw) // nw)
    if pk.lin:
        nr = rows or max(1, 200000 // nw)
        ch = np.array_split(np.arange(2 * nw), -(-2 * nw // nr))  # at most two chunk sizes
        r = r + jnp.concatenate([sum(_lin(md, g, b, pk.hh, e, qa, qb, s, jnp.asarray(ii)) for b in pk.lin)
                                 for ii in ch])
    for b in pk.cho:
        r = r + _cho(g, b, e, qa, qb)
    r = r * (1.0 - jnp.eye(2 * nw))  # same orbit, same band: cancels in dn
    if pk.cav is not None:
        up, dn = _cav(md, g, st, pk.cav)
        z = jnp.zeros((nw, nw))
        r = r + jnp.block([[z, jnp.diag(up / g.w[wi])], [jnp.diag(dn / g.w[wi]), z]])
    return r


def check_db(r, e, t):
    """Largest detailed balance residual |log(R_ij / R_ji) + (E_i - E_j) / t|, on the host."""
    r, e = np.asarray(r), np.asarray(e)
    de = e[:, None] - e[None, :]
    ma = (r > 0.0) & (r.T > 0.0)
    return np.max(np.abs(np.log(r[ma]) - np.log(r.T[ma]) + de[ma] / t))
