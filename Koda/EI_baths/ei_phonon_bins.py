"""Real one-phonon couplings and rates on a uniform energy grid.

Rows are destinations; columns are sources. States flatten as (alpha k, beta k).
amp is the squared orbital coupling prefactor. Rates include 2*pi, with hbar=1;
the volume/BZ normalization is absorbed in amp and the solver's normalized w.
The common imaginary phase is removed, but orbital signs are retained.
Only reciprocal, inversion-symmetric phonon dispersions are supported here.
"""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)


def om_const(q, w0=1.0):
    return jnp.full(jnp.shape(q)[:-1], w0, dtype=jnp.float64)


def om_ac(q, cs=1.0):
    return cs * jnp.linalg.norm(jnp.asarray(q), axis=-1)


def om_gap(q, w0=1.0, cs=1.0):
    return jnp.hypot(w0, om_ac(q, cs))


def g_orb(q, om, amp=1.0, lam=1.0, c=(1.0, 1.0)):
    """Diagonal orbital amplitudes, shape (2, ...); om is the physical frequency.

    g = sqrt(amp) * |q| / (sqrt(om) * (|q|**2 + lam**2)) * diag(c).
    The zero-frequency rigid translation at q=0 is omitted explicitly.
    Nonzero momenta with zero frequency are rejected by shell().
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    om = jnp.asarray(om, dtype=jnp.float64)
    q2 = jnp.sum(q * q, axis=-1)
    den = jnp.sqrt(jnp.where(om > 0, om, 1.0)) * (q2 + lam * lam)
    s = jnp.where(om > 0, jnp.sqrt(amp * q2) / den, 0.0)
    return jnp.stack((c[0] * s, c[1] * s))


def _rot(st, g, i, j):
    nk = st.e.shape[1]
    a = jnp.concatenate((st.u, st.v))
    b = jnp.concatenate((-st.v, st.u))
    ki, kj = (i % nk)[:, None], (j % nk)[None, :]
    return (g[0, ki, kj] * a[i, None] * a[None, j]
            + g[1, ki, kj] * b[i, None] * b[None, j])


def G_mat(st, g):
    """Return U_i.T @ diag(g_a, g_b) @ U_j in the solver's flattened order."""
    i = jnp.arange(st.e.size)
    return _rot(st, jnp.asarray(g), i, i)


def bose(om, t):
    """Stable Bose factor for positive frequencies, with a separate T=0 branch.

    Zero-frequency entries are inactive rigid translations, not Bose modes
    with a finite occupation. They return zero and never enter a delta mask.
    """
    om, t = jnp.asarray(om), jnp.asarray(t)
    x = om / jnp.where(t > 0, t, 1.0)
    den = -jnp.expm1(-x)
    return jnp.where((om > 0) & (t > 0),
                     jnp.exp(-x) / jnp.where(den > 0, den, 1.0), 0.0)


class Factors(NamedTuple):
    nb: Any
    dm: Any
    dp: Any
    em: Any
    ab: Any


def _fac(ix, jx, ws, h, t):
    de = ix[:, None] - jx[None, :]
    m = jnp.rint(ws / h).astype(jnp.int64)
    dm = ((m > 0) & (de == -m)).astype(jnp.float64) / h
    dp = ((m > 0) & (de == m)).astype(jnp.float64) / h
    nb = bose(ws, t)
    return Factors(nb, dm, dp, (nb + 1.0) * dm, nb * dp)


def bose_delta(ix, ws, h, t):
    """Return Bose, delta, emission and absorption factors as (2N, 2N) arrays.

    ws must already be on the transfer grid: use shell(), not raw frequencies.
    dm = delta_h(dE + ws); dp = delta_h(dE - ws).
    """
    ix = jnp.asarray(ix).reshape(-1)
    return _fac(ix, ix, jnp.tile(jnp.asarray(ws), (2, 2)), h, t)


def rate_mat(G, f):
    """Return emission, absorption, total rates; no quadrature or Pauli weights."""
    s = 2 * jnp.pi * jnp.square(G)
    s = s * (1.0 - jnp.eye(s.shape[0], dtype=s.dtype))
    rm, rp = s * f.em, s * f.ab
    return rm, rp, rm + rp


def shell(q, om_fn, h, snap=False):
    """Evaluate the dispersion and choose its explicit transfer-grid version.

    snap=False requires om/h to be integral (e.g. w0=M*h).
    snap=True rounds positive frequencies to the nearest multiple of h.
    Unresolved positive frequencies rounding to zero are rejected, not removed.
    """
    q = np.asarray(q, dtype=float)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    om = np.broadcast_to(np.asarray(om_fn(jnp.asarray(q)), dtype=float), q.shape[:-1]).copy()
    if not np.isfinite(om).all() or np.any(om < 0):
        raise ValueError("dispersion must give finite nonnegative frequencies")
    if np.any((om == 0) & (np.linalg.norm(q, axis=-1) > 1e-12)):
        raise ValueError("zero frequency at nonzero q is not supported")
    np.testing.assert_allclose(om, om.T, atol=1e-12, rtol=1e-12,
                               err_msg="dispersion must obey omega(q)=omega(-q)")
    m = np.rint(om / h).astype(np.int64)
    if np.any((om > 0) & (m == 0)):
        raise ValueError("positive phonon frequency unresolved: reduce h")
    ws = m * h
    if not snap and not np.allclose(om, ws, atol=1e-12 * h, rtol=1e-12):
        raise ValueError("frequency is off the energy grid: choose h or opt into snap=True")
    return om, ws


@dataclass(frozen=True)
class PhononBath:
    t: float
    k: Any
    ed: Any
    omega: Callable
    amp: float = 1.0
    lam: float = 1.0
    c: tuple = (1.0, 1.0)
    a0: float = 1.0
    snap: bool = False
    name: str = "phonon"


class Pack(NamedTuple):
    k: Any
    q: Any
    per: Any
    ed: Any
    h: Any
    g: Any
    om: Any
    ws: Any
    t: Any


def pack(bs):
    """Prepare static geometry and orbital couplings for ei_jax_2's bath hook."""
    from phonon_grid import mom_grid
    bs = tuple(bs)
    if not bs:
        raise ValueError("at least one bath is required")
    b0 = bs[0]
    k, ed = np.asarray(b0.k, float), np.asarray(b0.ed, float)
    if ed.ndim != 1 or ed.size < 2 or not np.isfinite(ed).all():
        raise ValueError("invalid fixed energy-bin edges")
    h = float(ed[1] - ed[0])
    if h <= 0:
        raise ValueError("energy-bin width must be positive")
    np.testing.assert_allclose(np.diff(ed), h, rtol=1e-12, atol=1e-14)
    q = mom_grid(b0, a0=b0.a0)
    gs, oms, wss, ts = [], [], [], []
    for b in tqdm(bs, desc="Phonon geometry", leave=False):
        if not np.array_equal(b.k, k) or not np.array_equal(b.ed, ed) or b.a0 != b0.a0:
            raise ValueError("all baths must share momenta, bin edges and a0")
        v = np.asarray((b.t, b.amp, b.lam, *b.c), dtype=float)
        if len(b.c) != 2 or not np.isfinite(v).all() or min(b.t, b.amp) < 0 or b.lam <= 0:
            raise ValueError("require T, amp >= 0, lam > 0 and two finite orbital factors")
        om, ws = shell(q, b.omega, h, snap=b.snap)
        gs.append(g_orb(q, om, b.amp, b.lam, b.c))
        oms.append(om)
        wss.append(ws)
        ts.append(b.t)
    return Pack(jnp.asarray(k), jnp.asarray(q), jnp.asarray(2 * np.pi / b0.a0),
                jnp.asarray(ed), jnp.asarray(h),
                jnp.stack(gs), jnp.asarray(np.stack(oms)),
                jnp.asarray(np.stack(wss)), jnp.asarray(ts))


def _idx(st, bp):
    e = st.e.reshape(-1)
    ix = jnp.searchsorted(bp.ed, e, side="right") - 1
    ok = jnp.all(jnp.isfinite(e) & (e >= bp.ed[0]) & (e < bp.ed[-1]))
    return ix, ok


def _pair(st, bp, i, j):
    ix, ok = _idx(st, bp)
    nk = st.e.shape[1]
    ki, kj = (i % nk)[:, None], (j % nk)[None, :]
    mask = (i[:, None] != j[None, :])

    def body(b, rr):
        G = _rot(st, bp.g[b], i, j)
        f = _fac(ix[i], ix[j], bp.ws[b, ki, kj], bp.h, bp.t[b])
        s = 2 * jnp.pi * G * G * mask
        r1 = s * (f.em + f.ab)
        r2 = s * (f.nb * f.dm + (f.nb + 1.0) * f.dp)
        return rr[0] + r1, rr[1] + r2

    z = jnp.zeros((i.size, j.size), dtype=jnp.float64)
    r1, r2 = jax.lax.fori_loop(0, bp.t.size, body, (z, z))
    return jnp.where(ok, r1, jnp.nan), jnp.where(ok, r2, jnp.nan)


def rates(st, bp):
    """JIT-compatible dense rate hook. Out-of-range energies return NaNs."""
    i = jnp.arange(st.e.size)
    return _pair(st, bp, i, i)[0]


def cur_block(st, n, w, bp, block):
    """JIT-compatible blocked current using the identical pair-rate kernel."""
    y, ww = n.reshape(-1), jnp.tile(w, 2)
    nm = y.size
    i = jnp.arange(nm)

    def body(b, ab):
        jj = b * block + jnp.arange(block)
        j = jnp.minimum(jj, nm - 1)
        r1, r2 = _pair(st, bp, i, j)
        wj = ww[j] * (jj < nm)
        return ab[0] + r1 @ (wj * y[j]), ab[1] + r2 @ (wj * (1.0 - y[j]))

    z = jnp.zeros_like(y)
    a, b = jax.lax.fori_loop(0, (nm + block - 1) // block, body, (z, z))
    return ((1.0 - y) * a - y * b).reshape(n.shape)


def parts(st, bp, b=0):
    """Expose one bath's separate factors for inspection on the host."""
    ix, ok = _idx(st, bp)
    if not bool(ok):
        raise ValueError("spectrum outside fixed bins; supply wider edges")
    eb = 0.5 * (bp.ed[ix] + bp.ed[ix + 1])
    G = G_mat(st, bp.g[b])
    f = bose_delta(ix, bp.ws[b], bp.h, bp.t[b])
    rm, rp, r = rate_mat(G, f)
    return dict(g=bp.g[b], G=G, om=bp.om[b], ws=bp.ws[b], ix=ix, eb=eb,
                nb=f.nb, dm=f.dm, dp=f.dp, em=f.em, ab=f.ab,
                rm=rm, rp=rp, r=r)


def audit(bd, st, bp, b=0, n=None):
    """Check momentum, energy exchange, number conservation and thermal balance.

    Checks use bin energies; physical detuning is reported separately.
    Random occupations are used for the conservation checks unless supplied.
    """
    x = {k: np.asarray(v) for k, v in parts(st, bp, b).items()}
    w = np.tile(np.asarray(bd.w, float), 2)
    e, eb, r = np.asarray(st.e).reshape(-1), x["eb"], x["r"]
    if not np.isfinite(r).all() or np.any(r < 0):
        raise AssertionError("rates must be finite and nonnegative")
    de, dx = eb[:, None] - eb[None, :], e[:, None] - e[None, :]
    ws, om = np.tile(x["ws"], (2, 2)), np.tile(x["om"], (2, 2))
    h, t = float(bp.h), float(bp.t[b])
    y = np.random.default_rng(7).uniform(0.1, 0.9, e.size) if n is None else np.asarray(n).reshape(-1)
    flux = w[:, None] * w[None, :] * y[None, :] * (1 - y[:, None])
    dn = (1 - y) * (r @ (w * y)) - y * (r.T @ (w * (1 - y)))
    je = np.sum(w * eb * dn)
    jp = np.sum(flux * ws * (x["rp"] - x["rm"]))
    pm, pp = x["rm"] > 0, x["rp"] > 0
    mm = np.max(np.abs((de + ws)[pm]), initial=0)
    mp = np.max(np.abs((de - ws)[pp]), initial=0)
    dm = np.max(np.abs((dx + om)[pm]), initial=0)
    dp = np.max(np.abs((dx - om)[pp]), initial=0)
    scale = max(float(np.max(r)), np.finfo(float).tiny)
    etol = 1e-11 * max(1.0, np.max(np.abs(eb)), np.max(ws))
    np.testing.assert_allclose([mm, mp], 0, atol=etol, rtol=0)
    np.testing.assert_allclose(np.sum(w * dn), 0, atol=1e-11 * scale, rtol=0)
    np.testing.assert_allclose(je, jp, atol=1e-11 * scale * max(1., np.max(np.abs(eb))), rtol=0)
    lim = h + np.max(np.abs(x["om"] - x["ws"]))
    if max(dm, dp) > lim + etol:
        raise AssertionError("physical energy detuning exceeds the bin error bound")
    k, q = np.asarray(bd.k), np.asarray(bp.q)
    z = k[:, None, :] - k[None, :, :] - q
    per = float(bp.per)
    km = float(np.max(np.abs(z / per - np.rint(z / per))))
    np.testing.assert_allclose(km, 0, atol=1e-11, rtol=0)
    db, eq, one = np.nan, np.nan, 0
    if t > 0:
        ma = (r > 0) & (r.T > 0)
        one = int(np.count_nonzero((r > 0) ^ (r.T > 0)))
        db = (float(np.max(np.abs(np.log(r[ma]) - np.log(r.T[ma]) + de[ma] / t)))
              if np.any(ma) else np.nan)
        f = np.asarray(jax.nn.sigmoid(-jnp.asarray(eb) / t))
        df = (1 - f) * (r @ (w * f)) - f * (r.T @ (w * (1 - f)))
        eq = float(np.max(np.abs(df)) / scale)
        if np.any(ma):
            np.testing.assert_allclose(db, 0, atol=1e-10, rtol=0)
        np.testing.assert_allclose(eq, 0, atol=1e-11, rtol=0)
    return dict(emission=int(pm.sum()), absorption=int(pp.sum()),
                bin_energy=max(mm, mp), physical_detuning=max(dm, dp),
                detuning_bound=lim, momentum_modulo=km,
                number=float(np.sum(w * dn)), energy_exchange=float(je - jp),
                db_log=db, one_way_entries=one, thermal_current_rel=eq)


def plot_bz(bd, vals, labs, a0=1.0, cmap="viridis", signed=False, xy="k"):
    """Plot arrays on the full Cartesian BZ; ordering of bd.k is arbitrary."""
    import matplotlib.pyplot as plt
    k = np.asarray(bd.k) * a0
    x, ix = np.unique(k[:, 0], return_inverse=True)
    y, iy = np.unique(k[:, 1], return_inverse=True)
    if x.size * y.size != len(k) or np.unique(iy * x.size + ix).size != len(k):
        raise ValueError("BZ plotting requires a full Cartesian momentum grid")
    vals = np.asarray(vals)
    nc = 1
    nr = (len(vals) + nc - 1) // nc
    h = 0.88 * nr
    fig, ax = plt.subplots(nr, nc, squeeze=False, figsize=(3.47412, h * 3.47412))
    vm = np.max(np.abs(vals))
    vl = -vm if signed else 0.0
    if vm == 0:
        vm = 1.0
        vl = -1.0 if signed else 0.0
    for j in tqdm(range(len(vals)), desc="BZ panels", leave=False):
        aa = ax.flat[j]
        z = np.empty((y.size, x.size))
        z[iy, ix] = vals[j]
        im = aa.pcolormesh(x, y, z, shading="nearest", cmap=cmap,
                          vmin=vl, vmax=vm, rasterized=True)
        im.set_edgecolor("face")
        cbar = plt.colorbar(im, ax=aa)
        cbar.set_label(labs[j], rotation=90, labelpad=5)
        aa.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
               xlabel=rf"${xy}_x a_0$", ylabel=rf"${xy}_y a_0$")
        aa.set_aspect("equal")
        aa.tick_params(labelsize=7)
        cbar.ax.tick_params(labelsize=7)
    for aa in tqdm(list(ax.flat)[len(vals):], desc="Unused panels", leave=False):
        aa.set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig


def plot_transfer(bd, st, x, h, set1_list):
    """Plot rate-weighted electronic energy transfers; this is not net heat flow."""
    import matplotlib.pyplot as plt
    e = np.asarray(st.e).reshape(-1)
    eb = np.asarray(x["eb"])
    de, dx = eb[:, None] - eb[None, :], e[:, None] - e[None, :]
    ww = np.tile(np.asarray(bd.w), 2)
    wt = ww[:, None] * ww[None, :]
    r = np.asarray(x["r"])
    ma = r > 0
    if not np.any(ma):
        raise ValueError("no allowed transitions: inspect the frequency sweep and matrix elements")
    dw = h / 4
    lo = int(np.floor(min(dx[ma].min(), de[ma].min()) / dw)) - 2
    hi = int(np.ceil(max(dx[ma].max(), de[ma].max()) / dw)) + 2
    ed = dw * (np.arange(lo, hi + 2) - 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(2 * 3.47412, 0.8 * 3.47412))
    for a, key in enumerate(tqdm(("rm", "rp"), desc="Energy transfers", leave=False)):
        rw = wt * np.asarray(x[key])
        y0, _ = np.histogram(de.ravel(), ed, weights=rw.ravel())
        y1, _ = np.histogram(dx.ravel(), ed, weights=rw.ravel())
        ax[a].stairs(y0, ed, color=set1_list[0], label=r"$\bar E_i-\bar E_j$")
        ax[a].stairs(y1, ed, color=set1_list[1], linestyle="--", label=r"$E_i-E_j$")
        om = np.asarray(x["om"])
        if np.allclose(om, om.flat[0]):
            v = (-1 if a == 0 else 1) * om.flat[0]
            ax[a].axvline(v, color="black", linestyle=":", linewidth=0.8,
                          label=r"$-\omega_0$" if a == 0 else r"$+\omega_0$")
        ax[a].set(xlim=(ed[0], ed[-1]), xlabel=r"$\Delta E$",
                   ylabel=r"$\sum_{ij\in I}w_i w_j R^{\rm em}_{ij}$" if a == 0
                   else r"$\sum_{ij\in I}w_i w_j R^{\rm abs}_{ij}$")
        ax[a].grid(alpha=0.3)
        ax[a].legend(fontsize=7)
    plt.tight_layout()
    plt.show()
    return fig
