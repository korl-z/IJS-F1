"""Nonthermal steady state on the shell grid with a bordered Newton solver.

Unknowns: GGE multipliers mu of every (branch, cell), n = 1 / (1 + exp(mu)),
the order parameter d, the imbalance m and one slack kap per conserved block.
Equations:
  balance   mu_i + log a_i - log b_i + kap_c = 0     (dn_i/dt = 0)
  gap       1 - V sum W <(n_beta - n_alpha) / 2E> = 0
  imbalance m - sum W <(u^2 - v^2)(n_alpha - n_beta)> = 0
  filling   log(sum_{i in c} W_i n_i) - log(N_c) = 0  (holes if c is nearly full)
with a_i = sum_j K_ij n_j (in) and b_i = sum_j K_ji (1 - n_j) (out).
Cell averages <.> use sub node occupations from a linear interpolation of mu
in energy (exact for thermal states). The slacks vanish at the solution
because particle number is conserved.
"""

import warnings
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

from .eq import solve_eq
from .geom import kernels
from .grid import auto_ns, make_shells
from .kernel import cells, gge_coef, mf_terms, rates, sub_grad, sub_mu

TINY = 1e-300


class Sol(NamedTuple):
    d: float
    m: float
    n: np.ndarray  # occupations, (2, ns)
    mu: np.ndarray  # GGE multipliers, (2, ns)
    e: np.ndarray  # cell energies, (2, ns)
    sc: np.ndarray  # cell centers in s
    w: np.ndarray  # cell weights
    err: float
    it: int
    ok: bool
    stable: bool | None
    rate: float | None  # slowest relaxation rate
    nblock: int
    branch: str  # 'ordered' or 'normal'
    hist: dict


@dataclass
class _Prob:
    sh: object
    band: object
    mf: object
    baths: tuple
    A: np.ndarray
    normal: bool
    wt: np.ndarray  # weights tiled over both branches
    act: np.ndarray = None  # cells with any coupling
    blk: np.ndarray = None  # block label of active cells, -1 otherwise
    nc: int = 0
    tgt: np.ndarray = None  # small population of each block (particles or holes)
    cap: np.ndarray = None  # number of states of each block
    form: np.ndarray = None  # True: tgt counts particles, False: holes
    kg: list = None  # group constraints, sum of +-tgt over each group
    mu0: np.ndarray = None  # frozen multipliers of inactive cells
    grp: list = None  # exchange groups of blocks
    ncons: int = 0  # number of truly conserved populations

    @property
    def nth(self):
        return 1 if self.normal else 2


def _state(pb, mu, d, m):
    cl = cells(pb.sh, pb.band, pb.mf, d, m)
    K = rates(cl, pb.baths, pb.A)
    n = expit(-mu)
    p = expit(mu)  # holes 1 - n, accurate in both tails
    a = K @ n
    b = K.T @ p
    return cl, K, n, a, b


def _blocks(pb, K, n, mu):
    """Blocks with fixed population in the inner problem.

    Blocks are the connected components of the intra branch coupling graph,
    so the alpha and beta populations are separate. Blocks joined by
    interband couplings form an exchange group; their split is found by the
    outer loop in _solve_split. Components of the full graph are truly
    conserved; their populations stay at the values of the initial state.
    """
    ns = K.shape[0] // 2
    g = K + K.T
    g = g > 1e-14 * max(g.max(), TINY)
    act = g.any(axis=1)
    gi = g.copy()
    gi[:ns, ns:] = False
    gi[ns:, :ns] = False
    _, lab = connected_components(gi, directed=False)
    _, full = connected_components(g, directed=False)
    ids = np.unique(lab[act])
    blk = np.full(lab.size, -1)
    for c, i in enumerate(ids):
        blk[(lab == i) & act] = c
    pb.act, pb.blk, pb.nc = act, blk, ids.size
    # populations as particles or holes, whichever is smaller, so that
    # nearly full blocks keep full relative precision
    sn = np.bincount(blk[act], weights=(pb.wt * expit(-mu))[act], minlength=pb.nc)
    sp = np.bincount(blk[act], weights=(pb.wt * expit(mu))[act], minlength=pb.nc)
    pb.cap = np.bincount(blk[act], weights=pb.wt[act], minlength=pb.nc)
    pb.form = sn <= sp
    pb.tgt = np.where(pb.form, sn, sp)
    pb.mu0 = mu.copy()
    fb = np.array([full[np.flatnonzero(blk == c)[0]] for c in range(pb.nc)])
    pb.grp = [np.flatnonzero(fb == f) for f in np.unique(fb) if np.sum(fb == f) > 1]
    sg = np.where(pb.form, 1.0, -1.0)
    pb.kg = [float(np.sum(sg[g] * pb.tgt[g])) for g in pb.grp]
    pb.ncons = np.unique(full[act]).size
    if pb.ncons > 1:
        warnings.warn(f"{pb.ncons} separately conserved populations (e.g. no "
                      "interband channel); they are fixed by the initial state")


def _unpack(pb, z):
    N = pb.wt.size
    mu = z[:N]
    d = 0.0 if pb.normal else z[N]
    m = z[N + pb.nth - 1]
    return mu, d, m, z[N + pb.nth:]


def _mf_rows(pb, cl, mu, grad=False):
    """Gap residual and imbalance target from GGE interpolated sub nodes.

    With grad=True also returns d(rd)/dmu and d(m - mt)/dmu, shape (2 ns,).
    """
    coef = gge_coef(cl)
    mus = sub_mu(mu.reshape(2, -1), coef)
    nsb = expit(-mus)
    gap, _, mt = mf_terms(pb.sh, cl, nsb)
    rd = 1.0 - pb.mf.v * gap
    if not grad:
        return rd, mt
    w = pb.sh.w[:, None] * pb.sh.ws
    dns = -nsb * expit(mus)
    g_rd = np.stack((w * cl.ies, -w * cl.ies)) * (pb.mf.v * dns)
    g_m = np.stack((-w * cl.xes, w * cl.xes)) * dns
    return rd, mt, sub_grad(g_rd, coef).ravel(), sub_grad(g_m, coef).ravel()


def _fill(pb, n, p):
    """Block population constraints in log form.

    log of the particle number for blocks less than half full, log of the
    hole number otherwise (pb.form). Linear in a uniform shift of mu (the block
    chemical potential), so multiplicative population changes are solved in
    one Newton step and tiny populations are enforced to relative precision.
    """
    ia = pb.act
    sn = np.bincount(pb.blk[ia], weights=(pb.wt * n)[ia], minlength=pb.nc)
    sp = np.bincount(pb.blk[ia], weights=(pb.wt * p)[ia], minlength=pb.nc)
    return np.where(pb.form, np.log(np.maximum(sn, TINY)),
                    np.log(np.maximum(sp, TINY))) - np.log(pb.tgt)


def _F(pb, z, parts=None):
    mu, d, m, kap = _unpack(pb, z)
    cl, K, n, a, b = parts if parts is not None else _state(pb, mu, d, m)
    act = pb.act
    ks = np.where(act, kap[np.maximum(pb.blk, 0)] if pb.nc else 0.0, 0.0)
    r = np.where(act, mu + np.log(np.maximum(a, TINY)) - np.log(np.maximum(b, TINY)) + ks,
                 mu - pb.mu0)
    rd, rm = _mf_rows(pb, cl, mu)
    fill = _fill(pb, n, expit(mu))
    mfr = [m - rm] if pb.normal else [rd, m - rm]
    return np.concatenate((r, mfr, fill))


def _jac(pb, z):
    mu, d, m, kap = _unpack(pb, z)
    parts = _state(pb, mu, d, m)
    cl, K, n, a, b = parts
    N, nth, act = mu.size, pb.nth, pb.act
    nz = z.size
    F0 = _F(pb, z, parts)
    J = np.zeros((nz, nz))
    dn = -n * expit(mu)

    # balance rows, mu columns
    sa = np.maximum(a, TINY)[:, None]
    sb = np.maximum(b, TINY)[:, None]
    blk = np.eye(N) + (K / sa + K.T / sb) * dn[None, :]
    J[:N, :N] = np.where(act[:, None], blk, np.eye(N))
    # slack columns
    for c in range(pb.nc):
        J[:N, N + nth + c] = (pb.blk == c).astype(float)

    # mean field rows, mu columns (through the sub node interpolation)
    _, _, g_rd, g_m = _mf_rows(pb, cl, mu, grad=True)
    if pb.normal:
        J[N, :N] = g_m
    else:
        J[N, :N] = g_rd
        J[N + 1, :N] = g_m
    # filling rows in log form, see _fill
    p = expit(mu)
    ia = pb.act
    sn = np.bincount(pb.blk[ia], weights=(pb.wt * n)[ia], minlength=pb.nc)
    sp = np.bincount(pb.blk[ia], weights=(pb.wt * p)[ia], minlength=pb.nc)
    for c in range(pb.nc):
        if pb.form[c]:
            row = pb.wt * dn / sn[c]
        else:
            row = -pb.wt * dn / sp[c]
        J[N + nth + c, :N] = np.where(pb.blk == c, row, 0.0)

    # d and m columns by central differences; the overlap kernel is C1, so
    # the step is near sqrt(eps) where truncation and roundoff balance
    for j in range(nth):
        k = N + j
        h = 1e-7 * max(abs(z[k]), 0.1)
        zp, zm = z.copy(), z.copy()
        zp[k] += h
        zm[k] -= h
        J[:, k] = (_F(pb, zp) - _F(pb, zm)) / (2.0 * h)
    return F0, J


MUCUT = 50.0  # cells with |mu| > MUCUT have n or 1 - n below 2e-22


def _err(pb, z, F):
    """Max residual over rows that can affect anything at double precision."""
    N = pb.wt.size
    rel = np.ones(F.size, dtype=bool)
    rel[:N] = np.abs(z[:N]) <= MUCUT
    return float(np.max(np.abs(F[rel])))


def _newton(pb, z, tol, nmax, hist, dmin, cap=40.0):
    """Newton with backtracking on a weighted |F|^2.

    Balance rows are weighted by sqrt(n (1 - n)) in the merit function and
    the mu step is clipped per cell, so deep tail cells (n ~ exp(-100) at low
    T) cannot block steps that fix the physically relevant cells. Rows of
    cells beyond MUCUT are excluded from the convergence test.
    Returns (z, status, err) with status 'ok', 'stall', 'maxit' or
    'collapse' (the order parameter fell below dmin).
    """
    N = pb.wt.size
    F, J = _jac(pb, z)
    for _ in range(nmax):
        err = _err(pb, z, F)
        hist["err"].append(err)
        if err < tol:
            return z, "ok", err
        try:
            dz = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, -F, rcond=None)[0]
        dz[:N] = np.clip(dz[:N], -cap, cap)
        s = 1.0
        if not pb.normal and dz[N] < -0.5 * z[N]:
            s = min(s, -0.5 * z[N] / dz[N])
        wr = np.ones(z.size)
        mu = z[:N]
        wr[:N] = np.sqrt(expit(-mu) * expit(mu))
        wr[:N] /= max(wr[:N].max(), TINY)
        f0 = np.sum((wr * F) ** 2)
        while s > 1e-6:
            z1 = z + s * dz
            F1 = _F(pb, z1)
            if np.all(np.isfinite(F1)) and np.sum((wr * F1) ** 2) < (1.0 - 1e-4 * s) * f0:
                break
            s *= 0.5
        hist["step"].append(s)
        if s <= 1e-6:
            # converged to the rounding floor if the residual is already tiny
            return z, ("ok" if err < 1e3 * tol else "stall"), err
        z = z1
        F, J = _jac(pb, z)
        if not pb.normal and z[N] < dmin:
            return z, "collapse", _err(pb, z, F)
    return z, "maxit", _err(pb, z, F)


def _logit(pb, s):
    """Block logits log(particles / holes) from small populations s."""
    t = np.where(pb.form, s, pb.cap - s)
    h = np.where(pb.form, pb.cap - s, s)
    return np.log(t) - np.log(h)


def _from_logit(pb, v):
    """Small populations from block logits, group constraints restored.

    Returns None if a population would become nonpositive.
    """
    s = np.where(pb.form, pb.cap * expit(v), pb.cap * expit(-v))
    sg = np.where(pb.form, 1.0, -1.0)
    for g, kg in zip(pb.grp, pb.kg):
        k = g[np.argmax(s[g])]
        s[k] += (kg - np.sum(sg[g] * s[g])) / sg[k]
    return s if np.all(s > 0) and np.all(s < pb.cap) else None


def _xflux(pb, z):
    """Half log ratio of in and out flux of each block, from the other blocks
    of its group only. Equals atanh(net / gross) and is nearly linear in the
    block logit, since in and out fluxes follow mass action."""
    mu, d, m, _ = _unpack(pb, z)
    cl, K, n, a, b = _state(pb, mu, d, m)
    p = expit(mu)
    r = np.zeros(pb.nc)
    for gi in pb.grp:
        ing = np.isin(pb.blk, gi)
        for c in gi:
            ic = pb.blk == c
            ot = ing & ~ic
            fin = p[ic] @ (K[np.ix_(ic, ot)] @ n[ot])
            fout = n[ic] @ (K[np.ix_(ot, ic)].T @ p[ot])
            r[c] = 0.5 * (np.log(max(fin, TINY)) - np.log(max(fout, TINY)))
    return r


def _solve_split(pb, z, tol, nmax, hist, dmin, nout=12, xtol=1e-9):
    """Inner Newton at fixed block populations, outer Newton on their split.

    The outer residual is the log ratio of in and out interband flux of every
    block in an exchange group, computed directly from the interband terms so
    that it stays well scaled even when interband rates are tiny. dz/dtgt follows
    from one solve with the inner Jacobian and gives a linear predictor.
    """
    tin = pb.tgt.copy()
    z, st, err = _newton(pb, z, tol, nmax, hist, dmin)
    if st != "ok" or not pb.grp:
        return z, st, err
    N, nth, nc = pb.wt.size, pb.nth, pb.nc
    ig = np.concatenate(pb.grp)
    rold, slow = np.inf, 0
    for _ in range(nout):
        r = _xflux(pb, z)
        rn = np.max(np.abs(r[ig]))
        if rn < xtol:
            return z, "ok", err
        # give up early if the split iteration does not contract
        slow = slow + 1 if rn > 0.5 * rold else 0
        if slow >= 2:
            break
        rold = rn
        F, J = _jac(pb, z)
        # dz / dtgt: the log form constraints have dF / dtgt = -1 / tgt
        E = np.zeros((z.size, nc))
        E[N + nth + np.arange(nc), np.arange(nc)] = 1.0 / pb.tgt
        S = np.linalg.solve(J, E)
        # Newton step in the block logits v = log(particles / holes)
        tt = np.where(pb.form, pb.tgt, pb.cap - pb.tgt)
        tv = tt * (pb.cap - tt) / pb.cap
        ds = np.where(pb.form, 1.0, -1.0) * tv  # dtgt / dv
        D = np.zeros((nc, nc))
        for c in ig:
            h = 1e-5 * ds[c]
            D[:, c] = (_xflux(pb, z + h * S[:, c]) - _xflux(pb, z - h * S[:, c])) / 2e-5
        dv = np.zeros(nc)
        for gi in pb.grp:
            M = np.vstack((D[np.ix_(gi, gi)], tv[gi]))
            rhs = np.concatenate((-r[gi], [0.0]))
            dv[gi] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        s = min(1.0, 2.0 / max(np.max(np.abs(dv)), TINY))
        t0, z0 = pb.tgt.copy(), z.copy()
        v0 = _logit(pb, t0)
        st = "fail"
        for _ in range(3):
            tn = _from_logit(pb, v0 + s * dv)
            if tn is not None:
                pb.tgt = tn
                # first order predictor in the logit variables
                z, st, err = _newton(pb, z0 + S @ (ds * s * dv), tol, nmax, hist, dmin)
                if st == "ok":
                    break
            s *= 0.5
        if st != "ok":
            pb.tgt = tin
            return z0, st, err
    pb.tgt = tin
    return z, "maxit", err


def _to_normal(pb, z):
    """Drop d from the unknowns and switch to the normal branch."""
    N = pb.wt.size
    pb.normal = True
    return np.concatenate((z[:N], z[N + 1:]))


def _path(baths, ts, lam):
    """Bath temperatures moved linearly from ts (lam = 0) to target (lam = 1).

    ts is one start temperature for all baths or one per bath.
    """
    ts = np.broadcast_to(np.asarray(ts, dtype=float), (len(baths),))
    return tuple(replace(b, t=float(t + lam * (b.t - t))) for b, t in zip(baths, ts))


def _homotopy(pb, z, baths, ts, tol, nmax, hist, dmin, fold=1.0 / 32):
    """Move the baths from ts to their targets, z must solve the start point.

    Adaptive steps with a secant predictor for z and the block split.
    Returns (z, status) with status 'ok', 'fold' (ordered branch lost: the
    step fell below fold) or 'fail'.
    """
    N = pb.wt.size
    lam, dl, st = 0.0, 1.0, "fail"
    prev = None  # last converged (lam, z, block logits)
    while lam < 1.0:
        lt = min(1.0, lam + dl)
        pb.baths = _path(baths, ts, lt)
        tg = pb.tgt.copy()
        zp = z
        if prev is not None and prev[1].size == z.size:
            q = (lt - lam) / (lam - prev[0])
            zp = z + q * (z - prev[1])
            v = _logit(pb, tg)
            tn = _from_logit(pb, v + np.clip(q * (v - prev[2]), -2.0, 2.0))
            pb.tgt = tg if tn is None else tn
            zp[N + pb.nth:] = 0.0
        z1, st, err = _solve_split(pb, zp, tol, nmax, hist, dmin)
        if st != "ok" and zp is not z:
            pb.tgt = tg
            z1, st, err = _solve_split(pb, z, tol, nmax, hist, dmin)
        if st != "ok":
            pb.tgt = tg
        if st == "ok":
            prev = (lam, z, _logit(pb, tg))
            z, lam = z1, lt
            hist["lam"].append(lam)
            dl = min(1.0, 2.0 * dl)
            continue
        dl *= 0.5
        if dl < fold and not pb.normal:
            hist["fold"] = lam
            return z, "fold"
        if dl < 1e-4:
            return z, "fail"
    return z, "ok"


def stability(pb, z):
    """Rightmost eigenvalue of dn/dt with instantaneous mean field.

    Linearize f = dn/dt at fixed (d, m), eliminate (d, m) through the mean
    field equations g(n, d, m) = 0 (Schur complement) and deflate the
    conserved directions. Returns the largest real part (negative = stable).
    """
    mu, d, m, _ = _unpack(pb, z)
    cl, K, n, a, b = _state(pb, mu, d, m)
    p = expit(mu)
    wt, nth, w = pb.wt, pb.nth, pb.sh.w
    fn = (-np.diag(a + b) + p[:, None] * K + n[:, None] * K.T) / wt[:, None]

    def fdot(th):
        cl1 = cells(pb.sh, pb.band, pb.mf, 0.0 if pb.normal else th[0], th[-1])
        K1 = rates(cl1, pb.baths, pb.A)
        return (p * (K1 @ n) - n * (K1.T @ p)) / wt

    def gmf(th):
        # mean field residuals at fixed cell occupations n
        dd = 0.0 if pb.normal else th[0]
        cl1 = cells(pb.sh, pb.band, pb.mf, dd, th[-1])
        _, dg, mt = mf_terms(pb.sh, cl1, expit(-sub_mu(mu.reshape(2, -1), gge_coef(cl1))))
        if pb.normal:
            return np.array([th[-1] - mt])
        return np.array([dd - pb.mf.v * dg, th[-1] - mt])

    th = np.array([m] if pb.normal else [d, m])
    fth = np.zeros((n.size, nth))
    gth = np.zeros((nth, nth))
    for j in range(nth):
        h = 1e-5 * max(abs(th[j]), 0.1)
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        fth[:, j] = (fdot(tp) - fdot(tm)) / (2 * h)
        gth[:, j] = (gmf(tp) - gmf(tm)) / (2 * h)
    # dg/dn = dg/dmu dmu/dn with dmu/dn = -1 / (n (1 - n))
    coef = gge_coef(cl)
    mus = sub_mu(mu.reshape(2, -1), coef)
    nsb = expit(-mus)
    dns = -nsb * expit(mus)
    wq = pb.sh.w[:, None] * pb.sh.ws
    inv = -1.0 / np.maximum(n * p, TINY)
    gn = np.zeros((nth, n.size))
    gm = np.stack((-wq * cl.xes, wq * cl.xes)) * dns
    gn[-1] = sub_grad(gm, coef).ravel() * inv
    if not pb.normal:
        gd = np.stack((wq * cl.uvs, -wq * cl.uvs)) * (pb.mf.v * dns)
        gn[0] = sub_grad(gd, coef).ravel() * inv
    S = fn - fth @ np.linalg.solve(gth, gn)

    # deflate conserved blocks and frozen cells
    s0 = 10.0 * max(np.max(np.abs(S)), 1.0)
    one = np.eye(n.size, dtype=bool)
    ixs = [pb.blk == c for c in range(pb.nc)] + [one[i] for i in np.flatnonzero(~pb.act)]
    for ix in ixs:
        lv = np.where(ix, wt, 0.0)
        S -= s0 * np.outer(ix / lv.sum(), lv)
    return float(np.max(np.linalg.eigvals(S).real))


def t_eff(baths):
    """Coupling weighted mean bath temperature."""
    a = np.array([b.amp for b in baths])
    t = np.array([b.t for b in baths])
    return float(np.sum(a * t) / np.sum(a))


def pair_chi(sh, band, mf, sol, dz=1e-6):
    """V chi_0 of a normal state sol: the gap equation at d -> 0.

    The ordered branch bifurcates from the normal one where this equals 1
    (n is even in d, so the occupations do not change at first order).
    > 1: the normal state is unstable towards excitonic order.
    """
    cl = cells(sh, band, mf, dz, sol.m)
    nsb = expit(-sub_mu(np.reshape(sol.mu, (2, -1)), gge_coef(cl)))
    return float(mf.v * mf_terms(sh, cl, nsb)[0])


def _start(sh, band, mf, ts, d, m, normal, equal=False):
    """Shell equilibrium used as the exact lam = 0 point of the homotopy.

    equal: all baths are at ts, so the steady state is the equilibrium at ts
    and ts is not lowered when the ordered equilibrium does not exist.
    """
    if not normal:
        for _ in range(1 if equal else 12):
            eq = solve_eq(sh, band, mf, ts, d, m)
            if eq.ok:
                return eq, ts, False
            ts *= 0.7
    eq = solve_eq(sh, band, mf, ts, 0.0, m, normal=True)
    return eq, ts, True


def solve_ness(sh, band, mf, baths, A=None, d=1.0, m=0.0, mu=None, t0=None,
               normal=False, tol=1e-11, nmax=25, stab=True, cache=None, dmin=1e-7,
               t_from=None, direct=False, fold=1.0 / 32):
    """Steady state of the bath driven EI on the shell grid.

    A: None (kernels are built, cached in cache if given), one Kern per bath,
    or the original constant dispersion pair_kernel array.
    mu given: Newton directly from (mu, d, m), e.g. for continuation. If that
    fails and t_from (bath temperatures at which (mu, d, m) is a solution)
    is given, the baths are moved from t_from to their targets. Otherwise a
    temperature homotopy from equilibrium is used: at T1 = T2 = t0 (default
    t_eff) the steady state is exactly the shell equilibrium. If the ordered
    branch is lost on the way, the normal branch (d = 0) is returned.
    direct (with mu): no homotopy from equilibrium, ok = False if the Newton
    from (mu, d, m) and the homotopy from t_from fail.
    fold: smallest homotopy step before the ordered branch counts as lost.
    """
    baths = tuple(baths)
    # one binned pair kernel per bath (shared between equal geometries)
    A = kernels(sh, band, baths, A, cache=cache)
    pb = _Prob(sh, band, mf, baths, A, bool(normal), np.tile(sh.w, 2))
    N = pb.wt.size
    hist = {"err": [], "step": [], "lam": []}

    def zvec(mu, d, m):
        th = [m] if pb.normal else [d, m]
        return np.concatenate((np.reshape(mu, -1), th, np.zeros(pb.nc)))

    ok = False
    if mu is not None:
        mu = np.reshape(mu, -1)
        _blocks(pb, _state(pb, mu, d, m)[1], expit(-mu), mu)
        z, st, err = _solve_split(pb, zvec(mu, d, m), tol, nmax, hist, dmin)
        if st == "collapse":
            z, st, err = _solve_split(pb, _to_normal(pb, z), tol, nmax, hist, dmin)
        ok = st == "ok"
        hist["lam"].append(1.0)
        if not ok and t_from is not None:
            # homotopy from the point that (mu, d, m) solves
            pb.normal = bool(normal)
            pb.baths = _path(baths, t_from, 0.0)
            _blocks(pb, _state(pb, mu, d, m)[1], expit(-mu), mu)
            z, st = _homotopy(pb, zvec(mu, d, m), baths, t_from, tol, nmax, hist, dmin,
                              fold)
            ok = st == "ok"

    if not ok and not (direct and mu is not None):
        ts = t_eff(baths) if t0 is None else float(t0)
        equal = all(b.t == ts for b in baths)
        eq, ts, nrm = _start(sh, band, mf, ts, d, m, normal, equal)
        pb.normal = nrm
        pb.baths = _path(baths, ts, 0.0)
        mu0 = eq.mu.reshape(-1)
        _blocks(pb, _state(pb, mu0, eq.d, eq.m)[1], eq.n.reshape(-1), mu0)
        z, st = _homotopy(pb, zvec(mu0, eq.d, eq.m), baths, ts, tol, nmax, hist, dmin,
                          fold)
        if st == "fold":
            # ordered branch ends (collapse or fold): solve the d = 0 branch
            warnings.warn(f"ordered branch lost at lam = {hist['fold']:.4g}, "
                          "returning the normal branch")
            sol = solve_ness(sh, band, mf, baths, A=A, m=m, t0=ts, normal=True,
                             tol=tol, nmax=nmax, stab=stab, dmin=dmin)
            sol.hist["fold"] = hist["fold"]
            return sol
        ok = st == "ok"
    pb.baths = baths

    # recheck the block structure at the solution
    mu, d, m, _ = _unpack(pb, z)
    cl, K, n, a, b = _state(pb, mu, d, m)
    old = (pb.act.copy(), pb.blk.copy())
    _blocks(pb, K, n, mu)
    if ok and not (np.array_equal(old[0], pb.act) and np.array_equal(old[1], pb.blk)):
        z, st, err = _solve_split(pb, zvec(mu, d, m), tol, nmax, hist, dmin)
        ok = st == "ok"
        mu, d, m, _ = _unpack(pb, z)
        cl, K, n, a, b = _state(pb, mu, d, m)

    lam = stability(pb, z) if (stab and ok) else None
    stable = None if lam is None else lam < 1e-9 * max(np.max(np.abs(K)), 1.0)
    ns = sh.ns
    return Sol(float(d), float(m), n.reshape(2, ns), mu.reshape(2, ns), cl.e,
               sh.sc, sh.w, _err(pb, z, _F(pb, z)), len(hist["err"]), ok,
               stable, None if lam is None else -lam, pb.ncons,
               "normal" if pb.normal else "ordered", hist)


def solve_auto(band, mf, baths, scale, tol=1e-5, ns=None, nref=5, cache=None, **kw):
    """Refine the shell grid until d and m change by less than tol.

    scale: smallest physical energy to resolve, e.g. min(d0, Tc, w0).
    Returns the finest solution and a dict with the refinement history and
    Richardson extrapolated d and m (the discretization error is O(h^2)).
    """
    ns = auto_ns(band, scale) if ns is None else ns
    rec = {"ns": [], "d": [], "m": [], "err": []}
    prev, sol = None, None
    for _ in range(nref):
        sh = make_shells(ns)
        if prev is not None:
            mu = np.array([np.interp(sh.sc, prev.sc, prev.mu[i]) for i in range(2)])
            kw.update(mu=mu, d=prev.d, m=prev.m, normal=prev.branch == "normal")
        sol = solve_ness(sh, band, mf, baths, cache=cache, **kw)
        rec["ns"].append(ns)
        rec["d"].append(sol.d)
        rec["m"].append(sol.m)
        rec["err"].append(sol.err)
        if prev is not None and sol.ok:
            r2 = (ns / rec["ns"][-2]) ** 2
            rec["d_ext"] = sol.d + (sol.d - prev.d) / (r2 - 1.0)
            rec["m_ext"] = sol.m + (sol.m - prev.m) / (r2 - 1.0)
            if abs(sol.d - prev.d) < tol and abs(sol.m - prev.m) < tol:
                break
        prev = sol
        ns = 2 * ns + 1
    return sol, rec


def to_k(sol, sh, k):
    """Occupations (2, nk) on arbitrary momenta k (nk, 2), e.g. bd.k."""
    k = np.asarray(k, dtype=float)
    return sol.n[:, sh.cell(np.cos(k[:, 0]) + np.cos(k[:, 1]))]


def sweep(sh, band, mf, baths_of, xs, A=None, d=1.0, m=0.0, normal=False,
          prefer="ordered", nfold=12, lfold=0.25, **kw):
    """Continuation along a 1d parameter list, e.g. one row of a phase map.

    baths_of(x) returns the baths at parameter x. Each point starts from the
    previous solution. After an ordered point: direct Newton (at most nfold
    iterations); with prefer = 'ordered' also a homotopy from the previous
    point (fold = lfold, so a lost branch is detected quickly). If that
    fails, the normal branch is kept if it is stable against order
    (pair_chi < 1), else the point is solved from equilibrium. After a
    normal point: the normal branch is continued; if it turns unstable
    (pair_chi > 1) the ordered branch is solved from equilibrium.
    prefer = 'ordered' follows the ordered branch as long as it exists;
    'normal' is faster but takes the normal state wherever it is stable,
    also where a distant ordered state coexists (hysteresis). Run the list
    in both directions to see hysteresis.
    """
    # kernels do not depend on temperature: build them once for the row
    A = kernels(sh, band, baths_of(xs[0]), A, cache=kw.get("cache"))

    def stable_normal(s):
        return s.ok and s.branch == "normal" and pair_chi(sh, band, mf, s) < 1.0

    out, prev, tp = [], None, None
    for x in xs:
        bs = baths_of(x)
        s = None
        if prev is not None and prev.ok and prev.branch == "ordered":
            # direct Newton, for 'ordered' then the homotopy from the last point
            tf = tp if prefer == "ordered" else None
            s = solve_ness(sh, band, mf, bs, A=A, mu=prev.mu, d=prev.d, m=prev.m,
                           t_from=tf, direct=True, **{**kw, "nmax": nfold, "fold": lfold})
            if not s.ok or (s.branch == "normal" and not stable_normal(s)):
                # ordered branch lost: stable normal state, else from equilibrium
                sn = s if (s.ok and s.branch == "normal") else solve_ness(
                    sh, band, mf, bs, A=A, mu=prev.mu, d=0.0, m=prev.m,
                    normal=True, direct=True, **kw)
                s = sn if stable_normal(sn) else solve_ness(
                    sh, band, mf, bs, A=A, d=d, m=m, **kw)
        elif prev is not None and prev.ok:
            s = solve_ness(sh, band, mf, bs, A=A, mu=prev.mu, d=0.0, m=prev.m,
                           normal=True, t_from=tp, **kw)
            if s.ok and not normal and pair_chi(sh, band, mf, s) > 1.0:
                s = solve_ness(sh, band, mf, bs, A=A, d=d, m=s.m, **kw)
        if s is None:
            s = solve_ness(sh, band, mf, bs, A=A, d=d, m=m, normal=normal, **kw)
        out.append(s)
        prev, tp = s, [b.t for b in bs]
    return out
