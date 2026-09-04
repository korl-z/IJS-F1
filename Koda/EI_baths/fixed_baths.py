"""Solve one two-bath state and save four diagnostic plots with Sacred."""

from __future__ import annotations

from pathlib import Path

import colormaps as cmaps
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from sacred import Experiment
from sacred.observers import FileStorageObserver

from utils.logger import logger, tqdm_bar
from utils.cosmetics import SET1_LIST, apply_plt_style
from utils.plotting_utils import (
    Plotter,
    add_legend,
    color_map,
    format_ax,
)

import EI.eff_T as et
import EI.ei_jax as ej
import EI.ei_unified as eu

CFG = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/config/config_test.yaml")
RUNS = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/runs")

# Edit these two ratios to select the plotted state.
T1_TC = 2.0
T2_TC = 0.2
N_CONT = 10
SHOW = False


with CFG.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["selected_state"] = {
    "t1_tc": T1_TC,
    "t2_tc": T2_TC,
    "n_cont": N_CONT,
    "show": SHOW,
}
cfg["effective_temperature"] = {
    "dt": 0.05,
    "mix": [0.15, 0.15],
    "tol": 1.0e-6,
    "nmax": 20000,
    "chk": 10,
    "mf": False,
    "prog": True,
    **cfg.get("effective_temperature", {}),
}

RUNS.mkdir(parents=True, exist_ok=True)
ex = Experiment("selected_state_plots")
ex.add_config(cfg)
ex.observers.append(FileStorageObserver.create(str(RUNS)))


def fd(z):
    """Stable Fermi-Dirac function for a dimensionless argument."""
    return 1.0 / (np.exp(np.clip(z, -500.0, 500.0)) + 1.0)


def fd_t(e, t):
    """Fermi-Dirac function with a zero-temperature limit."""
    if t <= 0.0:
        return np.where(e < 0.0, 1.0, np.where(e > 0.0, 0.0, 0.5))
    return fd(e / t)


def make_baths(t1, t2, bp):
    """Build the two bath objects without repeating the rate setup."""
    b1 = ej.gam_db(t=t1, name="bath 1", **bp)
    b2 = ej.gam_db(t=t2, name="bath 2", **bp)
    return b1, b2


def k_edges(x):
    """Convert uniform cell centers to pcolormesh edges."""
    x = np.asarray(x)
    xm = 0.5 * (x[:-1] + x[1:])
    return np.r_[x[0] - 0.5 * (x[1] - x[0]), xm,
                 x[-1] + 0.5 * (x[-1] - x[-2])]


def k_path(bd):
    """Return the Gamma-X-M-Gamma path on the square momentum grid."""
    kg = np.asarray(bd.k).reshape(bd.shape + (bd.dim,))
    ig = np.unravel_index(np.argmin(np.linalg.norm(kg, axis=-1)), bd.shape)
    i0 = int(ig[0])
    ip = int(np.argmin(kg[:, 0, 0]))

    p1 = np.column_stack((np.arange(i0, ip - 1, -1),
                          np.full(i0 - ip + 1, i0)))
    p2 = np.column_stack((np.full(i0 - ip, ip),
                          np.arange(i0 - 1, ip - 1, -1)))
    ii = np.arange(ip + 1, i0 + 1)
    p3 = np.column_stack((ii, ii))
    ij = np.vstack((p1, p2, p3))

    kk = kg[ij[:, 0], ij[:, 1]]
    x = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(kk, axis=0), axis=1))]
    ix = len(p1) - 1
    im = len(p1) + len(p2) - 1
    return ij, x, x[[0, ix, im, -1]]


def take_path(z, ij, sh):
    """Take one or two flattened fields along a grid path."""
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(sh)
        return z[ij[:, 0], ij[:, 1]]
    z = z.reshape((z.shape[0],) + sh)
    return z[:, ij[:, 0], ij[:, 1]]


def color_line(ax, x, y, c, w, cmap, norm, ls="-", label=None):
    """Draw a line colored by occupation and weighted by orbital character."""
    xy = np.column_stack((x, y)).reshape(-1, 1, 2)
    sg = np.concatenate((xy[:-1], xy[1:]), axis=1)
    cc = 0.5 * (c[:-1] + c[1:])
    ww = 0.6 + 1.5 * 0.5 * (w[:-1] + w[1:])
    lc = LineCollection(sg, cmap=cmap, norm=norm, linewidths=ww,
                        linestyles=ls, zorder=4)
    lc.set_array(cc)
    ax.add_collection(lc)
    ax.update_datalim(np.column_stack((x, y)))
    ax.plot([], [], color="0.25", ls=ls, lw=1.4, label=label)
    return lc


def solve_state(bd, p, eq, op, r1, r2, tc, d0, m0, nc):
    """Continue from the cold equilibrium state to one selected bath pair."""
    t1, t2 = r1 * tc, r2 * tc
    mu = 0.5 * p.v
    bc = op["bath"]
    bp = {**bc["pars"], "rate": getattr(ej, bc["rate"]), "mu": mu}
    es = {**eq["solve"], "prog": False}
    se = eu.solve_eq(bd, p, t=t2, d=max(d0, 1.0e-8), m=m0, **es)

    # n, d, m = se.n.copy(), se.d, se.m
    # ds = max(0.02 * d0, 1.0e-3)
    st = None
    nc = 1 if np.isclose(r1, r2) else max(2, int(nc))
    # xa = np.linspace(r2, r1, nc)
    # for x in tqdm_bar(xa, desc="T1 continuation"):
    #     bs = make_baths(x * tc, t2, bp)
    #     st = ej.solve_open(bd, p, n, bs, d=d, m=m, **op["solve"])
    #     n, d, m = st.n.copy(), max(abs(st.d), ds), st.m
    bs = make_baths(t1, t2, bp)
    st = ej.solve_open(
        bd, p, se.n.copy(), bs,
        d=se.d,
        m=se.m,
        **op["solve"],
    )

    if not st.ok:
        logger.warning("Open-state solver did not converge: err = %.3e", st.err)
    return st, make_baths(t1, t2, bp), t1, t2, mu


def solve_teff(bd, p, bs, st, t1, t2, mu, op, ep):
    """Fit the zero-power effective thermal state."""
    kw = dict(ep)
    kw.setdefault("mode", op["solve"].get("mode", "block"))
    kw.setdefault("block", op["solve"].get("block", 512))
    sf = et.solve_eff(bd, p, bs, beta0=2.0 / max(t1 + t2, 1.0e-12),
                      d=st.d, m=st.m, mu=mu, **kw)
    if not sf.ok:
        logger.warning("Effective-temperature solver did not converge: err = %.3e",
                       sf.err)
    return sf


def plot_dispersion(pt, bd, st, mu):
    """Plot bare, Hartree, and diagonal bands on Gamma-X-M-Gamma."""
    ij, x, xt = k_path(bd)
    eb = take_path(np.stack((bd.ea, bd.eb)), ij, bd.shape)
    eh = take_path(np.stack((st.st.eah, st.st.ebh)), ij, bd.shape) - mu
    ed = take_path(st.st.e, ij, bd.shape) - mu
    nq = take_path(st.n, ij, bd.shape)
    u = take_path(st.st.u, ij, bd.shape)
    v = take_path(st.st.v, ij, bd.shape)
    norm = Normalize(0.0, 1.0)
    sm = ScalarMappable(norm=norm, cmap=cmaps.vanimo)

    with pt.figure(name="dispersion", h=0.78) as q:
        a = q.ax
        a.plot(x, eb[0], color=SET1_LIST[0], ls="--", lw=0.9,
               label=r"$\varepsilon_{a\mathbf{k}}$")
        a.plot(x, eb[1], color=SET1_LIST[1], ls="--", lw=0.9,
               label=r"$\varepsilon_{b\mathbf{k}}$")
        a.plot(x, eh[0], color=SET1_LIST[0], lw=1.1,
               label=r"$\varepsilon^{\mathrm{H}}_{a\mathbf{k}}-V/2$")
        a.plot(x, eh[1], color=SET1_LIST[1], lw=1.1,
               label=r"$\varepsilon^{\mathrm{H}}_{b\mathbf{k}}-V/2$")
        color_line(a, x, ed[0], nq[0], u**2, cmaps.vanimo, norm,
                   label=r"$E_{\alpha\mathbf{k}}-V/2$")
        color_line(a, x, ed[1], nq[1], v**2, cmaps.vanimo, norm, ls="--",
                   label=r"$E_{\beta\mathbf{k}}-V/2$")
        a.axvline(xt[1], color="0.75", lw=0.7)
        a.axvline(xt[2], color="0.75", lw=0.7)

        yy = np.r_[eb.ravel(), eh.ravel(), ed.ravel()]
        dy = 0.04 * max(np.ptp(yy), 1.0)
        format_ax(a, x=x, xlabel=r"$\mathbf{k}$",
                  ylabel=r"$\varepsilon_{\mathbf{k}}$",
                  ylim=(yy.min() - dy, yy.max() + dy),
                  xticks=xt,
                  xticklabels=[r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"])
        add_legend(a, top=True, ncol=3, fontsize=7)
        q.colorbar(sm, r"$n_{\lambda\mathbf{k}}$", ax=a)
        q.layout((0, 0, 1, 0.86))


def plot_energy(pt, st, sf, t1, t2, mu, d0):
    """Plot NESS occupations and thermal curves against energy."""
    en = (np.asarray(st.st.e) - mu) / d0
    ef = (np.asarray(sf.st.e) - mu) / d0
    nn = np.asarray(st.n)
    x = np.linspace(min(en.min(), ef.min()), max(en.max(), ef.max()), 600)

    with pt.figure(name="occupations_energy", h=0.68) as q:
        a = q.ax
        a.plot(x, fd_t(x * d0, t1), color=SET1_LIST[0], ls="--", lw=1.0,
               label=r"$f_{\mathrm{FD}}(E,T_1)$")
        a.plot(x, fd_t(x * d0, t2), color=SET1_LIST[1], ls="--", lw=1.0,
               label=r"$f_{\mathrm{FD}}(E,T_2)$")
        a.plot(x, fd(sf.beta * x * d0), color=SET1_LIST[4], lw=1.4,
               label=r"$f_{\mathrm{FD}}(E,T_{\mathrm{eff}})$")
        a.scatter(en[0], nn[0], color=SET1_LIST[2], s=3, alpha=0.55,
                  edgecolors="none", label=r"$n_{\alpha\mathbf{k}}^{\mathrm{NESS}}$")
        a.scatter(en[1], nn[1], color=SET1_LIST[3], s=3, alpha=0.55,
                  edgecolors="none", label=r"$n_{\beta\mathbf{k}}^{\mathrm{NESS}}$")
        a.axvline(0.0, color="0.65", ls=":", lw=0.7)
        format_ax(a, x=x, xlabel=r"$(E-V/2)/\Delta_0$", ylabel=r"$n$",
                  ylim=(0.0, 1.0))
        add_legend(a, top=True, ncol=3, fontsize=7)
        q.layout((0, 0, 1, 0.84))


def plot_xi(pt, bd, p, eq, st, sf, t1, t2, d0):
    """Plot the same occupation comparison against the normal-state xi."""
    es = {**eq["solve"], "prog": False}
    s1 = eu.solve_eq(bd, p, t=t1, d=max(d0, 1.0e-3), m=st.m, **es)
    s2 = eu.solve_eq(bd, p, t=t2, d=max(d0, 1.0e-3), m=st.m, **es)
    eh = np.stack((np.asarray(st.st.eah), np.asarray(st.st.ebh)))
    hf = np.stack((np.asarray(sf.st.eah), np.asarray(sf.st.ebh)))
    xn = np.abs(0.5 * (eh[0] - eh[1])) / d0
    xf = np.abs(0.5 * (hf[0] - hf[1])) / d0
    x = np.linspace(0.0, max(xn.max(), xf.max()), 600)
    q1 = np.sqrt((x * d0)**2 + s1.d**2)
    q2 = np.sqrt((x * d0)**2 + s2.d**2)
    qf = np.sqrt((x * d0)**2 + sf.d**2)
    nn = np.asarray(st.n)

    with pt.figure(name="occupations_xi", h=0.68) as q:
        a = q.ax
        a.plot(x, fd_t(q1, t1), color=SET1_LIST[0], ls="--", lw=1.0,
               label=r"$f_{\mathrm{FD}}(\xi,T_1)$")
        a.plot(x, fd_t(q2, t2), color=SET1_LIST[1], ls="--", lw=1.0,
               label=r"$f_{\mathrm{FD}}(\xi,T_2)$")
        a.plot(x, fd(sf.beta * qf), color=SET1_LIST[4], lw=1.4,
               label=r"$f_{\mathrm{FD}}(\xi,T_{\mathrm{eff}})$")
        a.scatter(xn, nn[0], color=SET1_LIST[2], s=3, alpha=0.5,
                  edgecolors="none", label=r"$n_{\alpha\mathbf{k}}^{\mathrm{NESS}}$")
        a.scatter(xn, 1.0 - nn[1], color=SET1_LIST[3], s=3, alpha=0.5,
                  edgecolors="none", label=r"$1-n_{\beta\mathbf{k}}^{\mathrm{NESS}}$")
        format_ax(a, x=x, xlabel=r"$|\xi_{\mathbf{k}}|/\Delta_0$",
                  ylabel=r"$n_{\mathrm{ex}}$", ylim=(0.0, 0.52))
        add_legend(a, top=False, ncol=1, fontsize=7)
        q.layout((0, 0, 1, 0.84))


def plot_bz(pt, bd, st):
    """Plot alpha and beta occupations over the two-dimensional zone."""
    kg = np.asarray(bd.k).reshape(bd.shape + (bd.dim,))
    kx = k_edges(kg[:, 0, 0] / np.pi)
    ky = k_edges(kg[0, :, 1] / np.pi)
    nf = np.asarray(st.n).reshape((2,) + bd.shape)

    with pt.figure(name="occupations_bz", shape=(1, 2), h=0.62,
                   sharex=True, sharey=True) as q:
        im0 = color_map(q.ax[0], kx, ky, nf[0].T, cmap=cmaps.lipari,
                        vmin=0.0, vmax=1.0)
        im1 = color_map(q.ax[1], kx, ky, nf[1].T, cmap=cmaps.lipari,
                        vmin=0.0, vmax=1.0)
        q.ax[0].text(0.05, 0.93, r"$n_{\alpha\mathbf{k}}$",
                     transform=q.ax[0].transAxes, ha="left", va="top")
        q.ax[1].text(0.05, 0.93, r"$n_{\beta\mathbf{k}}$",
                     transform=q.ax[1].transAxes, ha="left", va="top")

        format_ax(q.ax[0], x=kx, xlabel=r"$k_x/\pi$", ylabel=r"$k_y/\pi$",
                  grid=False)
        format_ax(q.ax[1], x=kx, xlabel=r"$k_x/\pi$", grid=False)
        q.ax[0].set_ylim(ky[0], ky[-1])
        q.ax[1].set_ylim(ky[0], ky[-1])
        q.ax[0].set_aspect("equal")
        q.ax[1].set_aspect("equal")
        q.colorbar(im1, r"$n_{\lambda\mathbf{k}}$", ax=q.ax[-1],
                   external=True)


@ex.automain
def main(_run, model, equilibrium, open_system, selected_state,
         effective_temperature):
    """Run the selected-state calculation and create the four artifacts."""
    apply_plt_style()
    _run.add_resource(str(CFG))

    nk = model["nk"]
    bd = eu.tb_2d(nk["scan"], **model["band"]["pars"])
    br = eu.tb_2d(nk["reference"], **model["band"]["pars"])
    p = eu.MFPars(**model["mean_field"])

    z0 = {**equilibrium["initial"], "t": 0.0}
    s0 = eu.solve_eq(br, p, **z0, **equilibrium["solve"])
    d0 = max(abs(float(s0.d)), 1.0e-12)
    tc = float(eu.critical_temperature(br, p, **equilibrium["critical"]))
    r1 = float(selected_state["t1_tc"])
    r2 = float(selected_state["t2_tc"])

    st, bs, t1, t2, mu = solve_state(
        br, p, equilibrium, open_system, r1, r2, tc, d0, s0.m,
        selected_state["n_cont"],
    )
    sf = solve_teff(br, p, bs, st, t1, t2, mu, open_system,
                    effective_temperature)
    pt = Plotter(run=_run, show=bool(selected_state["show"]), close=True)

    logger.info("Delta_0 = %.8f", d0)
    logger.info("T_c = %.8f", tc)
    logger.info("T1/T_c = %.6g, T2/T_c = %.6g", r1, r2)
    logger.info("Delta = %.8f, m = %.8f", st.d, st.m)
    logger.info("beta_eff = %.8f, T_eff/T_c = %.8f", sf.beta, sf.t / tc)

    _run.info["state"] = {
        "delta_0": d0,
        "tc": tc,
        "t1_tc": r1,
        "t2_tc": r2,
        "delta": float(st.d),
        "m": float(st.m),
        "open_ok": bool(st.ok),
        "open_err": float(st.err),
        "beta_eff": float(sf.beta),
        "t_eff_tc": float(sf.t / tc),
        "eff_ok": bool(sf.ok),
        "eff_err": float(sf.err),
        "power": float(sf.power),
    }
    _run.log_scalar("delta", float(st.d))
    _run.log_scalar("open_error", float(st.err))
    _run.log_scalar("t_eff_tc", float(sf.t / tc))
    _run.log_scalar("power", float(sf.power))

    plot_dispersion(pt, br, st, mu)
    plot_energy(pt, st, sf, t1, t2, mu, d0)
    plot_xi(pt, br, p, equilibrium, st, sf, t1, t2, d0)
    plot_bz(pt, br, st)
    logger.info("Saved four plots in Sacred run %s", _run._id)
