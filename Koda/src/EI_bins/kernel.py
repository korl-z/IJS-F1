"""Cell energies, coherence factors and rates at a given (d, m).

States are flattened as (alpha cells, beta cells), alpha is the upper branch.
K[i, j] is the pair flux coefficient for j -> i, so that
W_i dn_i/dt = (1 - n_i) sum_j K[i, j] n_j - n_i sum_j K[j, i] (1 - n_j).
"""

from typing import NamedTuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from .geom import Kern


class Cells(NamedTuple):
    e: np.ndarray  # cell energy centers, (2, ns)
    hw: np.ndarray  # cell energy widths, (2, ns)
    ma: np.ndarray  # alpha orbital second moments, (ns, 2, 2)
    mb: np.ndarray  # beta orbital second moments, (ns, 2, 2)
    es: np.ndarray  # sub node energies, (2, ns, nsub)
    uvs: np.ndarray  # u v at the sub nodes, (ns, nsub)
    xes: np.ndarray  # u^2 - v^2 at the sub nodes
    ies: np.ndarray  # 1 / (2 E) at the sub nodes


def _qp(s, band, mf, d, m):
    """xi, eta and sqrt(xi^2 + d^2) at shell values s."""
    ea, eb = band.bare(s)
    xi = 0.5 * (ea - eb) - 0.5 * mf.h * m
    et = 0.5 * (ea + eb) + 0.5 * mf.h * (mf.n - 1.0)
    return xi, et, np.hypot(xi, d)


def cells(sh, band, mf, d, m, hmin=1e-12):
    """Cell energies, coherence moments and sub node quantities at (d, m).

    Center and width are the DOS weighted mean and sqrt(12 var) of E over the
    cell, so both are smooth in (d, m). The factor nsub^2 / (nsub^2 - 1)
    removes the bias of the discrete variance of a linear E(s).
    """
    xi, et, ek = _qp(sh.ss, band, mf, d, m)

    def avg(f):
        return np.sum(sh.ws * f, axis=-1)

    ns2 = sh.ss.shape[1] ** 2
    cor = 12.0 * ns2 / max(ns2 - 1.0, 1.0)
    es = np.stack((et + ek, et - ek))
    e = avg(es)
    hw = np.sqrt(np.maximum(cor * avg((es - e[..., None]) ** 2), hmin**2))

    ok = ek > 0.0
    ep = np.where(ok, ek, 1.0)
    xr = np.where(ok, xi / ep, 1.0)
    uv = np.where(ok, d / (2.0 * ep), 0.0)
    a, b, c = avg(0.5 * (1.0 + xr)), avg(0.5 * (1.0 - xr)), avg(uv)
    ma = np.stack((np.stack((a, -c), -1), np.stack((-c, b), -1)), -2)
    mb = np.stack((np.stack((b, c), -1), np.stack((c, a), -1)), -2)
    ie = np.where(ok, 0.5 / ep, 0.0)
    return Cells(e, hw, ma, mb, es, uv, xr, ie)


def gge_coef(cl):
    """Linear interpolation of mu in energy inside each cell.

    mu_sub = c0 mu_I + cm mu_{I-1} + cp mu_{I+1} (per branch), with the slope
    dmu/dE fitted to the neighbouring cells. Exact for thermal states, where
    mu = (E - mc) / T is linear in E.
    """
    e = cl.e
    dp = np.zeros_like(e)
    dm = np.zeros_like(e)
    dp[:, :-1] = e[:, 1:] - e[:, :-1]
    dm[:, 1:] = e[:, :-1] - e[:, 1:]
    s2 = dp**2 + dm**2
    s2 = np.where(s2 > 0.0, s2, 1.0)
    de = cl.es - e[..., None]
    cp = de * (dp / s2)[..., None]
    cm = de * (dm / s2)[..., None]
    return 1.0 - cp - cm, cm, cp


def sub_mu(mu, coef):
    """Sub node multipliers (2, ns, nsub) from cell multipliers (2, ns)."""
    c0, cm, cp = coef
    mup = np.zeros_like(mu)
    mum = np.zeros_like(mu)
    mup[:, :-1] = mu[:, 1:]
    mum[:, 1:] = mu[:, :-1]
    return c0 * mu[..., None] + cm * mum[..., None] + cp * mup[..., None]


def sub_grad(g, coef):
    """Chain rule: d/dmu_cell from d/dmu_sub values g (2, ns, nsub)."""
    c0, cm, cp = coef
    out = np.sum(g * c0, axis=-1)
    out[:, :-1] += np.sum(g * cm, axis=-1)[:, 1:]
    out[:, 1:] += np.sum(g * cp, axis=-1)[:, :-1]
    return out


def mf_terms(sh, cl, nsb):
    """Mean field sums from sub node occupations nsb (2, ns, nsub).

    Returns gap = sum W <(n_b - n_a) / 2E> (gap equation 1 = V gap),
    dg = sum W <u v (n_b - n_a)> (d = V dg) and
    mt = sum W <(u^2 - v^2)(n_a - n_b)> (m = mt).
    """
    w = sh.w[:, None] * sh.ws
    dn = nsb[1] - nsb[0]
    return (np.sum(w * cl.ies * dn), np.sum(w * cl.uvs * dn),
            -np.sum(w * cl.xes * dn))


def coh2(mi, mj, c):
    """Cell pair average of |U_I^T c U_J|^2 for one branch pair."""
    x = np.einsum("ab,iac,cd->ibd", c, mi, c)
    return np.einsum("ibd,jbd->ij", x, mj)


def trap(y, a, b):
    """Density at y of the sum of two centered uniforms with widths a, b."""
    return np.clip(0.5 * (a + b) - np.abs(y), 0.0, np.minimum(a, b)) / (a * b)


def trap_cdf(y, a, b):
    """Cumulative distribution of the trapezoid, exactly 0 and 1 outside."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)

    def left(y):
        t = np.clip(y + 0.5 * (lo + hi), 0.0, 0.5 * (lo + hi))
        return np.where(t < lo, t * t / (2.0 * lo * hi), 0.5 * lo / hi + (t - lo) / hi)

    return np.where(y <= 0.0, left(y), 1.0 - left(-y))


def line(y, a, b, cmin=0.0):
    """Energy overlap kernel of a cell pair with a smoothing box of width c.

    Trapezoid convolved with a box of width c = max((a + b) / 2, cmin):
    normalized, even in y, C1 in (y, a, b) and exactly zero outside its
    support. cmin keeps rates smooth on the grid resolution scale, also for
    the narrow cells at the gap edge.
    """
    y, a, b = np.broadcast_arrays(y, a, b)
    c = np.maximum(0.5 * (a + b), cmin)
    out = np.zeros(y.shape)
    ix = np.abs(y) < 0.5 * (a + b + c)  # support, exactly zero outside
    y, a, b, c = y[ix], a[ix], b[ix], c[ix]
    out[ix] = (trap_cdf(y + 0.5 * c, a, b) - trap_cdf(y - 0.5 * c, a, b)) / c
    return out


def bose(w, t):
    return 1.0 / np.expm1(w / t) if t > 0 else 0.0


# Backend generic pieces (xp = numpy or jax.numpy), shared with jax_core.

PI = np.pi
T3, W3 = leggauss(3)
T24, W24 = leggauss(24)


def cdf_xp(xp, y, a, b):
    """trap_cdf for any array backend."""
    lo, hi = xp.minimum(a, b), xp.maximum(a, b)

    def left(y):
        t = xp.clip(y + 0.5 * (lo + hi), 0.0, 0.5 * (lo + hi))
        return xp.where(t < lo, t * t / (2.0 * lo * hi), 0.5 * lo / hi + (t - lo) / hi)

    return xp.where(y <= 0.0, left(y), 1.0 - left(-y))


def line_xp(xp, y, a, b, cmin):
    """line for any array backend (numpy uses the masked version)."""
    if xp is np:
        return line(y, a, b, cmin)
    c = xp.maximum(0.5 * (a + b), cmin)
    v = (cdf_xp(xp, y + 0.5 * c, a, b) - cdf_xp(xp, y - 0.5 * c, a, b)) / c
    return xp.where(xp.abs(y) < 0.5 * (a + b + c), v, 0.0)


def bose_xp(xp, w, t):
    """Bose factor N(w) for w > 0, zero for t = 0."""
    if xp is np and np.ndim(t) == 0 and t <= 0:
        return np.zeros_like(w)
    return xp.where(t > 0.0, 1.0 / xp.expm1(w / xp.maximum(t, 1e-300)), 0.0)


def delta_geo(xp, x, ha, hb, cmin, om):
    """Geometric part of a sharp line at om: the overlap kernel at x - om."""
    return line_xp(xp, x - om, ha, hb, cmin)


def delta_brk(xp, le, om, t):
    """Sharp line bracket (1 + N) L(x - om) + N L(x + om)."""
    n = bose_xp(xp, om, t)
    return (1.0 + n) * le + n * le.T


def spec_line(xp, e, om, gam):
    """Damped oscillator line of width gam at om, normalized on e > 0."""
    z = 2.0 * xp.arctan(om / gam) / PI
    d1 = (e - om) ** 2 + gam**2
    d2 = (e + om) ** 2 + gam**2
    return xp.where(e > 0.0, 4.0 * gam * e * om / (PI * d1 * d2 * z), 0.0)


def _spec_nodes(xp, x, ha, hb, cmin, om, gam):
    """Nodes e and weights (times B L) for pairs of equal shaped x, ha, hb."""
    sh = x.shape + (24,)
    c = xp.maximum(0.5 * (ha + hb), cmin)
    s = 0.5 * (ha + hb + c)
    lo = xp.maximum(x - s, 0.0)
    hi = xp.maximum(x + s, 0.0)

    def F(e):
        return 0.5 + xp.arctan((e - om) / gam) / PI

    pts = [x - 0.5 * (p * ha + q * hb + r * c)
           for p in (1, -1) for q in (1, -1) for r in (1, -1)]
    pts.append(om + 0.0 * x)
    P = xp.sort(xp.clip(xp.stack(pts, -1), lo[..., None], hi[..., None]), axis=-1)
    U = F(P)
    u0, du = U[..., :-1], U[..., 1:] - U[..., :-1]
    ub = (u0[..., None] + du[..., None] * (0.5 * (T3 + 1.0))).reshape(sh)
    wb = (du[..., None] * (0.5 * W3)).reshape(sh)
    ul, uh = F(lo), F(hi)
    un = ul[..., None] + (uh - ul)[..., None] * (0.5 * (T24 + 1.0))
    wn = (uh - ul)[..., None] * (0.5 * W24)
    broad = (gam >= s / 8.0)[..., None]
    u = xp.where(broad, ub, un)
    wq = xp.where(broad, wb, wn)
    e = om + gam * xp.tan(PI * (u - 0.5))
    z = 2.0 * xp.arctan(om / gam) / PI
    br = 4.0 * e * om / (z * ((e + om) ** 2 + gam**2))  # B / Cauchy density
    L = line_xp(xp, x[..., None] - e, ha[..., None], hb[..., None], cmin)
    return e, wq * br * L


def spec_geo(xp, x, ha, hb, cmin, om, gam, tmin=np.inf):
    """Quadrature nodes e and weights (times B L) of the spectral bracket.

    B is the damped oscillator line of width gam centred at om, normalized
    on e > 0. The integral over e of B(e) L(x - e) uses Gauss-Legendre in
    the Cauchy mapped variable u = 1/2 + arctan((e - om) / gam) / pi, which
    concentrates nodes on the line. Broad lines (gam >= s / 8, s the kernel
    half support) split the support at the 8 kernel breakpoints and om
    (8 pieces x 3 nodes, exact for the piecewise quadratic kernel); narrow
    lines use one 24 point rule. Only e > 0 contributes.
    The numpy path treats pairs far from the line (|x - om| > 5 (s + gam))
    with a 3 point rule matching the kernel moments up to fourth order, if
    their energy spread is also small against tmin, the lowest temperature
    of the baths using these nodes (the Bose factor must be smooth).
    """
    x = x + 0.0 * ha + 0.0 * hb
    ha = ha + 0.0 * x
    hb = hb + 0.0 * x
    if xp is not np:
        return _spec_nodes(xp, x, ha, hb, cmin, om, gam)
    c = np.maximum(0.5 * (ha + hb), cmin)
    s = 0.5 * (ha + hb + c)
    # far: nodes x, x +- h with h^2 = 3 var, var = (a^2 + b^2 + c^2) / 12
    h = np.sqrt(0.25 * (ha**2 + hb**2 + c**2))
    far = (np.abs(x - om) > 5.0 * (s + gam)) & (x - s > 0.0) & (h < 0.1 * tmin)
    near = ~far & (x + s > 0.0)
    e = np.ones(x.shape + (24,))
    wl = np.zeros(x.shape + (24,))
    for k, (dx, wk) in enumerate(((0.0, 2.0 / 3.0), (-1.0, 1.0 / 6.0), (1.0, 1.0 / 6.0))):
        ek = x + dx * h
        e[..., k] = np.where(far, ek, 1.0)
        wl[..., k] = np.where(far, wk * spec_line(np, ek, om, gam), 0.0)
    if near.any():
        en, wn = _spec_nodes(np, x[near], ha[near], hb[near], cmin, om, gam)
        e[near] = en
        wl[near] = wn
    return e, wl


def spec_brk(xp, geo, t):
    """Spectral bracket: int B (1 + N) L(x - e) + [int B N L(x - e)]^T."""
    e, wl = geo
    # B N stays finite as e -> 0; zero length pieces put nodes at e = 0
    n = bose_xp(xp, xp.maximum(e, 1e-200), t)
    return xp.sum(wl * (1.0 + n), axis=-1) + xp.sum(wl * n, axis=-1).T


def rates(cl, baths, kerns):
    """Pair flux coefficients K (2 ns, 2 ns) summed over baths.

    kerns: one Kern per bath (geom.kernels), or the original constant
    dispersion pair kernel array. For every frequency bin m the downhill
    direction (source above destination) uses 2 pi amp |C|^2 A_m times the
    sharp line bracket (gam = 0) or the spectral bracket (gam > 0). The
    uphill direction follows from grid detailed balance,
    K[j, i] = K[i, j] exp(-(E_j - E_i) / T), so T1 = T2 gives exactly
    Fermi-Dirac occupations on the cell centers.
    """
    if isinstance(kerns, np.ndarray):
        kerns = tuple(Kern(kerns[None] / b.w0, np.array([b.w0]), 0.0) for b in baths)
    e = cl.e.reshape(-1)
    hw = cl.hw.reshape(-1)
    x = e[None, :] - e[:, None]
    ha, hb = hw[:, None], hw[None, :]
    cf = hw.max()
    m2 = (cl.ma, cl.mb)
    K = np.zeros_like(x)
    geo, cohs, tiles = {}, {}, {}
    tmin = {}
    for bt, kn in zip(baths, kerns):
        t = bt.t if bt.t > 0 else np.inf
        tmin[(id(kn), bt.gam)] = min(tmin.get((id(kn), bt.gam), np.inf), t)
    for bt, kn in zip(baths, kerns):
        if bt.c not in cohs:
            cm = bt.cm
            cohs[bt.c] = np.block([[coh2(m2[i], m2[j], cm) for j in range(2)]
                                   for i in range(2)])
        cmin = max(cf, kn.dw)
        p = np.zeros_like(x)
        for m in range(kn.om.size):
            key = (id(kn), m, bt.gam)
            if key not in geo:
                # shared by all baths with the same kernel and line width
                geo[key] = (spec_geo(np, x, ha, hb, cmin, kn.om[m], bt.gam,
                                     tmin[(id(kn), bt.gam)])
                            if bt.gam > 0 else delta_geo(np, x, ha, hb, cmin, kn.om[m]))
                tiles[key] = np.tile(kn.A[m], (2, 2))
            br = (spec_brk(np, geo[key], bt.t) if bt.gam > 0
                  else delta_brk(np, geo[key], kn.om[m], bt.t))
            p += tiles[key] * br
        p *= 2.0 * np.pi * bt.amp * cohs[bt.c]
        if bt.t > 0:
            up = p.T * np.exp(np.minimum(x, 0.0) / bt.t)
        else:
            up = np.zeros_like(p)
        K += np.where(x >= 0.0, p, up)
    np.fill_diagonal(K, 0.0)
    return K
