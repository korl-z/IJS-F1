"""Rate matrix R[dest, src] on the kinetic states (alpha, beta) x g.wi. Numpy.

R is the C4v average over the images of the source, so the solvers use
a = R (w n), b = R^T (w (1 - n)), dn = (1 - n) a - n b with the orbit weights w.
"""

import numpy as np
from scipy.special import expit


def therm(de, t):
    """Omega N(Omega) for Omega = E_dest - E_src: absorption w N(w), emission w (1 + N(w))."""
    w = np.abs(de)
    x = w / np.where(t > 0.0, t, 1.0)
    bw = np.where(x > 1.0e-8, w / np.expm1(np.clip(x, 1.0e-8, 700.0)), t)
    return np.where(t > 0.0, bw, 0.0) + np.maximum(-de, 0.0)


def kern(om, w0, gam):
    """Bracket of eq. 286: 4 gam Om int_w0^inf (w + w0^2 / w) / (d1 d2) dw."""
    z4 = (om**2 + gam**2) ** 2
    a = np.arctan((om - w0) / gam) + np.arctan((om + w0) / gam)
    l = np.log(((om - w0) ** 2 + gam**2) * ((om + w0) ** 2 + gam**2) / w0**4)
    return (1.0 + w0**2 * (om**2 - gam**2) / z4) * a + gam * om * w0**2 / z4 * l


def _toy(p, e, qa, qb, bd):
    """Sum of the toy baths, rows p = (t, wc, orb, mu, prod, kaa, kab, kba, kbb)."""
    de = e[:, None] - e[None, :]
    bi, bj = bd[:, None], bd[None, :]
    r = np.zeros_like(de)
    for t, wc, orb, mu, prod, kaa, kab, kba, kbb in p:
        kap = np.where(bi == 0, np.where(bj == 0, kaa, kab), np.where(bj == 0, kba, kbb))
        q = np.where(orb == 1.0, qa, np.where(orb == 2.0, qb, 1.0))
        f = np.where(prod > 0.0, expit((mu - e) / t)[:, None] * expit((e - mu) / t)[None, :],
                     expit(-de / t))
        r = r + kap * np.exp(-((de / wc) ** 2)) * f * (q[:, None] * q[None, :]) ** 2
    return r


def _look(tab, cen, iq, w, gam, hh):
    """Line tables at energy w and transfer iq, linear in the sinh grid index."""
    mm = (tab.shape[-1] - 1) // 2
    z = np.clip(np.arcsinh((w - cen[:, iq]) / gam) / hh + mm, 0.0, 2.0 * mm)
    i0 = np.minimum(z.astype(int), 2 * mm - 1)
    f = z - i0
    c = np.arange(tab.shape[0]).reshape(-1, *([1] * (i0.ndim - 1)))
    return (1.0 - f) * tab[c, iq, i0] + f * tab[c, iq, i0 + 1]


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
    tv = _look(tab, cen, iq, np.abs(de).reshape(nr, 2, nw, 1), gam, hh)  # (rows, nr, 2, nw, 8)
    aa, bb = qa[ii][:, None] * qa[None, :], qb[ii][:, None] * qb[None, :]
    if tab.shape[0] == 1:  # phonon: |sin k - sin k'|^2, L and T summed
        v2 = np.sum((g.sn[kt] - g.sn[ks]) ** 2, -1)[:, None]
        sm = (2.0 * (c1a * aa + c1b * bb)) ** 2 * np.mean(v2 * tv[0], -1).reshape(nr, -1)
    else:  # photon: h = sin(k' + q / 2), q in the first BZ
        qc = (qi + g.nk // 2) % g.nk - g.nk // 2
        h = g.s2[(2 * ks + qc) % (2 * g.nk)]
        hx, hy = h[..., 0][:, None], h[..., 1][:, None]
        x1 = np.mean(hx * hx * tv[0] + hy * hy * tv[1] + hx * hy * tv[2], -1).reshape(nr, -1)
        si, sj = s[ii][:, None], s[None, :]
        eab = md.gap - 2.0 * md.ta * si + 2.0 * md.tb * sj  # eq. 233
        eba = -md.gap - 2.0 * md.tb * si + 2.0 * md.ta * sj
        c2 = dip * (eab * qa[ii][:, None] * qb[None, :] + eba * qb[ii][:, None] * qa[None, :])
        sm = (2.0 * (c1a * aa + c1b * bb)) ** 2 * x1 + c2**2 * np.mean(tv[3], -1).reshape(nr, -1)
    return amp * sm * therm(de, t)


def _cho(g, b, e, qa, qb):
    """One chord bath: R = amp C^2 Omega N(Omega) L(I, J, |Omega|) / (w_I w_J).

    L = 1 / (2 pi) int q dq F(Omega, omega_q) S(I, J, q), F the Lorentzian of eq. 272.
    gam = 0: F -> 2 pi delta(|Omega| - cs q) / (cs q)^2, so L = S(q*) / (cs^2 |Omega|),
    q* = |Omega| / cs, zero beyond the Debye radius.
    """
    par, S, dq = b
    t, amp, gam, tpa, tpb, cs, w0, qd = par
    nw, nq = g.wi.shape[0], S.shape[-1]
    q = dq * np.arange(nq)
    om = np.hypot(w0, cs * q)
    wq = np.where(q <= qd, q * dq / (2.0 * np.pi), 0.0)
    gs = np.where(gam > 0.0, gam, 1.0)
    de = e[:, None] - e[None, :]
    i, j = np.arange(nw)[:, None], np.arange(nw)[None, :]
    blk = []
    for a in (0, 1):  # band blocks, all share the table
        row = []
        for c in (0, 1):
            w = np.abs(de[a * nw:(a + 1) * nw, c * nw:(c + 1) * nw])
            f = 8.0 * gs / (((w[..., None] - om) ** 2 + gs**2) * ((w[..., None] + om) ** 2 + gs**2))
            z = np.minimum(w / (cs * dq), nq - 1.0)
            l0 = np.minimum(z.astype(int), nq - 2)
            sh = ((1.0 - (z - l0)) * S[i, j, l0] + (z - l0) * S[i, j, l0 + 1]) / (cs**2 * np.maximum(w, 1.0e-300))
            ok = (w > 0.0) & (w <= cs * np.minimum(qd, (nq - 1) * dq))
            row.append(np.where(gam > 0.0, np.sum(S * f * wq, -1), np.where(ok, sh, 0.0)))
        blk.append(row)
    ww = np.concatenate((g.w[g.wi], g.w[g.wi]))
    c1 = 2.0 * (tpa * qa[:, None] * qa[None, :] + tpb * qb[:, None] * qb[None, :])
    return amp * c1**2 * np.block(blk) * therm(de, t) / (ww[:, None] * ww[None, :])


def _cav(md, g, st, p):
    """Vertical rates up (beta_k -> alpha_k) and dn (alpha_k -> beta_k), eq. 286, 287."""
    om = (st.e[0] - st.e[1])[g.wi]
    cc = g.hv[g.wi] * (2.0 * (md.ta - md.tb) * st.u[g.wi] * st.v[g.wi]) ** 2
    up, dn = np.zeros_like(om), np.zeros_like(om)
    for t, w0, gam, amp, c in p:
        j = amp / (2.0 * np.pi * c**2) * cc * kern(om, w0, gam)
        nb = np.where(t > 0.0, 1.0 / np.maximum(np.expm1(om / np.where(t > 0.0, t, 1.0)), 1.0e-300), 0.0)
        up, dn = up + j * nb, dn + j * (1.0 + nb)
    return up, dn


def rates(md, g, st, pk, rows=None):
    """R[dest, src] of all baths at the state st. rows: target rows per chunk (memory)."""
    wi = g.wi
    nw = wi.shape[0]
    e = st.e[:, wi].reshape(-1)
    qa = np.concatenate((st.u[wi], st.v[wi]))  # a orbital weight of (alpha, beta)
    qb = np.concatenate((-st.v[wi], st.u[wi]))
    s = np.concatenate((g.s[wi], g.s[wi]))
    r = np.zeros((2 * nw, 2 * nw))
    if pk.toy is not None:
        r = r + _toy(pk.toy, e, qa, qb, np.arange(2 * nw) // nw)
    if pk.lin:
        nr = rows or max(1, 100000 // nw)
        ch = np.array_split(np.arange(2 * nw), -(-2 * nw // nr))
        r = r + np.concatenate([sum(_lin(md, g, b, pk.hh, e, qa, qb, s, ii) for b in pk.lin)
                                for ii in ch])
    for b in pk.cho:
        r = r + _cho(g, b, e, qa, qb)
    r = r * (1.0 - np.eye(2 * nw))  # same orbit, same band: cancels in dn
    if pk.cav is not None:
        up, dn = _cav(md, g, st, pk.cav)
        z = np.zeros((nw, nw))
        r = r + np.block([[z, np.diag(up / g.w[wi])], [np.diag(dn / g.w[wi]), z]])
    return r


def check_db(r, e, t):
    """Largest detailed balance residual |log(R_ij / R_ji) + (E_i - E_j) / t|."""
    de = e[:, None] - e[None, :]
    ma = (r > 0.0) & (r.T > 0.0)
    return np.max(np.abs(np.log(r[ma]) - np.log(r.T[ma]) + de[ma] / t))
