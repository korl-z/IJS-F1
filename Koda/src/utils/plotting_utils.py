# import matplotlib.pyplot as plt
# import numpy as np
# import yaml
# from pathlib import Path
# from tempfile import TemporaryDirectory

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
# import colormaps as cmaps

# from tqdm.notebook import tqdm

# def plot_phase_diag(
#     da, d0, tn, config=None, OUT=None, name="phase_diag", save=False, overwrite=False
# ):
#     z = da / d0

#     h = 0.85

#     fig, ax = plt.subplots(figsize=(3.47412, h * 3.47412))

#     im = ax.pcolormesh(tn, tn, z, shading="auto", cmap=cmaps.lipari, vmin=0.0)
#     im.set_edgecolor("face")

#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label(r"$\Delta/\Delta_0$", rotation=90, labelpad=5)

#     # probe points
#     r = np.arange(config["ratio_sweep"]["r_min"], config["ratio_sweep"]["r_max"] + config["ratio_sweep"]["r_step"], config["ratio_sweep"]["r_step"])

#     # ax.scatter(
#     #     config["ratio_sweep"]["x0"] * np.sqrt(r),
#     #     config["ratio_sweep"]["x0"] / np.sqrt(r),
#     #     color="green",
#     # )

#     ax.set_xlabel(r"$T_1/T_c$")
#     ax.set_ylabel(r"$T_2/T_c$")
#     ax.set_xlim(tn.min(), tn.max())
#     ax.set_ylim(tn.min(), tn.max())

#     plt.tight_layout()

#     if save:
#         if OUT is None:
#             raise ValueError("OUT must be provided when save=True")

#         OUT = Path(OUT)
#         OUT.mkdir(parents=True, exist_ok=True)
        
#         if not overwrite and (plot_file.exists()):
#             raise FileExistsError(f"{name} already exists in {OUT}")
        
#         plot_file = OUT / f"{name}.pdf"
#         fig.savefig(plot_file, bbox_inches="tight")

#         print("Saved:", plot_file)

#         plt.show()
#         # plt.close(fig)


# def plot_nalpha_beta(
#     naa,
#     nba,
#     nea,
#     neb,
#     r,
#     ks,
#     kt,
#     kl,
#     config,
#     OUT=None,
#     name="nalpha_beta",
#     save=False,
#     overwrite=False,
# ):
#     x0 = config["ratio_sweep"]["x0"]
#     ra = r

#     norm = Normalize(vmin=ra.min(), vmax=ra.max())
#     sm = ScalarMappable(norm=norm, cmap=cmaps.guppy)
#     sm.set_array([])

#     h = 1.15
#     fig, ax = plt.subplots(2, 1, figsize=(3.47412, h * 3.47412), sharex=True)

#     for q in tqdm(range(r.size), desc="Plotting"):
#         cl = cmaps.guppy(norm(ra[q]))

#         ax[0].plot(ks, naa[q], color=cl, linewidth=1.3)
#         ax[1].plot(ks, nba[q], color=cl, linewidth=1.3)

#     ax[0].plot(
#         ks,
#         nea,
#         color="black",
#         linestyle="--",
#         linewidth=1.5,
#         label=rf"eq, $T/T_c={x0:.1f}$",
#     )
#     ax[1].plot(
#         ks,
#         neb,
#         color="black",
#         linestyle="--",
#         linewidth=1.5,
#         label=rf"eq, $T/T_c={x0:.1f}$",
#     )

#     ax[0].set_ylabel(r"$n_{\alpha\mathbf{k}}$")
#     ax[1].set_ylabel(r"$n_{\beta\mathbf{k}}$")
#     ax[1].set_xlabel(r"$\mathbf{k}$")

#     for a in ax:
#         a.set_xlim(ks[0], ks[-1])
#         a.set_xticks(kt)
#         a.set_xticklabels(kl)
#         a.grid(alpha=0.3)
#         a.legend(loc="best", fontsize=7)

#     cax = inset_axes(
#         ax[1],
#         width="3%",
#         height="211%",
#         loc="lower left",
#         bbox_to_anchor=(1.02, 0, 1, 1),
#         bbox_transform=ax[1].transAxes,
#         borderpad=0,
#     )

#     cbar = fig.colorbar(sm, cax=cax)
#     cbar.set_label(r"$T_1/T_2$", rotation=90, labelpad=5)

#     plt.tight_layout()

#     if save:
#         if OUT is None:
#             raise ValueError("OUT must be provided when save=True")

#         OUT = Path(OUT)
#         OUT.mkdir(parents=True, exist_ok=True)

#         plot_file = OUT / f"{name}.pdf"

#         if not overwrite and (plot_file.exists()):
#             raise FileExistsError(f"{name} already exists in {OUT}")

#         fig.savefig(plot_file, bbox_inches="tight")

#         print("Saved:", plot_file)

#         plt.show()
#         # plt.close(fig)


# def plot_nalpha_beta_bz(
#     nfa, r, q0, OUT=None, name="nalpha_beta_bz", save=False, overwrite=False
# ):
#     if q0 < 0 or q0 >= r.size:
#         raise IndexError("q0 is outside the ratio sweep")

#     nf = nfa[q0]
#     nk1 = nf.shape[-1]

#     ke = np.linspace(-1.0, 1.0, nk1 + 1)

#     h = 0.62
#     fig, ax = plt.subplots(
#         1, 2, figsize=(3.47412, h * 3.47412), sharex=True, sharey=True
#     )

#     im0 = ax[0].pcolormesh(
#         ke, ke, nf[0].T, shading="flat", cmap=cmaps.lipari, vmin=0.0, vmax=1.0
#     )
#     im0.set_edgecolor("face")

#     im1 = ax[1].pcolormesh(
#         ke, ke, nf[1].T, shading="flat", cmap=cmaps.lipari, vmin=0.0, vmax=1.0
#     )
#     im1.set_edgecolor("face")

#     ax[0].text(
#         0.05,
#         0.93,
#         r"$n_{\alpha\mathbf{k}}$",
#         transform=ax[0].transAxes,
#         color="white",
#         ha="left",
#         va="top",
#     )

#     ax[1].text(
#         0.05,
#         0.93,
#         r"$n_{\beta\mathbf{k}}$",
#         transform=ax[1].transAxes,
#         color="k",
#         ha="left",
#         va="top",
#     )

#     ax[1].text(
#         0.95,
#         0.93,
#         rf"$T_1/T_2={r[q0]:.2f}$",
#         transform=ax[1].transAxes,
#         color="k",
#         ha="right",
#         va="top",
#     )
#     ax[0].tick_params(which="both", direction="in")
#     ax[1].tick_params(which="both", direction="in")

#     for a in ax:
#         a.set_xlim(ke[0], ke[-1])
#         a.set_ylim(ke[0], ke[-1])
#         a.set_aspect("equal")
#         a.set_xlabel(r"$k_x/\pi$")

#     ax[0].set_ylabel(r"$k_y/\pi$")

#     cax = inset_axes(
#         ax[1],
#         width="3%",
#         height="100%",
#         loc="lower left",
#         bbox_to_anchor=(1.02, 0, 1, 1),
#         bbox_transform=ax[1].transAxes,
#         borderpad=0,
#     )

#     cbar = fig.colorbar(im1, cax=cax)
#     cbar.set_label(r"$n_{\lambda\mathbf{k}}$", rotation=90, labelpad=5)

#     plt.tight_layout()

#     if save:
#         if OUT is None:
#             raise ValueError("OUT must be provided when save=True")

#         OUT = Path(OUT)
#         OUT.mkdir(parents=True, exist_ok=True)

#         plot_file = OUT / f"{name}.pdf"

#         if not overwrite and (plot_file.exists()):
#             raise FileExistsError(f"{name} already exists in {OUT}")

#         fig.savefig(plot_file, bbox_inches="tight")

#         print("Saved:", plot_file)

#         plt.show()
#         # plt.close(fig)


# def plot_energies_uv(
#     eaa,
#     eba,
#     ua,
#     va,
#     eah,
#     ebh,
#     ra,
#     ks,
#     kt,
#     kl,
#     OUT=None,
#     name="energies_uv",
#     save=False,
#     overwrite=False,
#     Ec=None,
# ):
#     norm = Normalize(vmin=ra.min(), vmax=ra.max())
#     sm = ScalarMappable(norm=norm, cmap=cmaps.guppy)
#     sm.set_array([])

#     h = 1.15
#     fig, ax = plt.subplots(2, 1, figsize=(3.47412, h * 3.47412), sharex=True)

#     for q in tqdm(range(ra.size), desc="Plotting"):
#         cl = cmaps.guppy(norm(ra[q]))

#         ax[0].plot(ks, eaa[q], color=cl, linestyle="-", linewidth=1.2)
#         ax[0].plot(ks, eba[q], color=cl, linestyle="--", linewidth=1.2)

#         ax[1].plot(ks, ua[q], color=cl, linestyle="-", linewidth=1.2)
#         ax[1].plot(ks, va[q], color=cl, linestyle="--", linewidth=1.2)

#     ax[0].plot(ks, eah[0], color="black", linestyle="-", linewidth=0.5)
#     ax[0].plot(ks, ebh[0], color="black", linestyle="-", linewidth=0.5)

#     ax[0].plot([], [], color="black", linestyle="-", label=r"$E_{\alpha\mathbf{k}}$")
#     ax[0].plot([], [], color="black", linestyle="--", label=r"$E_{\beta\mathbf{k}}$")

#     ax[1].plot([], [], color="black", linestyle="-", label=r"$u_{\mathbf{k}}$")
#     ax[1].plot([], [], color="black", linestyle="--", label=r"$v_{\mathbf{k}}$")

#     ax[0].set_ylabel(r"$E_{\lambda\mathbf{k}}/|t_a|$")
#     ax[1].set_ylabel(r"$u_{\mathbf{k}},\,v_{\mathbf{k}}$")
#     ax[1].set_xlabel(r"$\mathbf{k}$")

#     ax[0].tick_params(which="both", direction="in")
#     ax[1].tick_params(which="both", direction="in")

#     if Ec is not None:
#         ax[0].axhline(
#             Ec,
#             xmin=0.0,
#             xmax=1.0,
#             color="k",
#             linestyle="-.",
#             linewidth=0.5,
#         )

#     for a in ax:
#         a.set_xlim(ks[0], ks[-1])
#         a.set_xticks(kt)
#         a.set_xticklabels(kl)
#         a.grid(alpha=0.3)
#         a.legend(loc="best", ncol=2, fontsize=8)

#     cax = inset_axes(
#         ax[1],
#         width="3%",
#         height="208%",
#         loc="lower left",
#         bbox_to_anchor=(1.02, 0, 1, 1),
#         bbox_transform=ax[1].transAxes,
#         borderpad=0,
#     )

#     cbar = fig.colorbar(sm, cax=cax)
#     cbar.set_label(r"$T_1/T_2$", rotation=90, labelpad=5)

#     plt.tight_layout()

#     if save:
#         if OUT is None:
#             raise ValueError("OUT must be provided when save=True")

#         OUT = Path(OUT)
#         OUT.mkdir(parents=True, exist_ok=True)

#         plot_file = OUT / f"{name}.pdf"

#         if not overwrite and (plot_file.exists()):
#             raise FileExistsError(f"{name} already exists in {OUT}")

#         fig.savefig(plot_file, bbox_inches="tight")

#         print("Saved:", plot_file)

#         plt.show()
#         # plt.close(fig)


# def newfig(nrows=1, ncols=1, W=3.47412, h=0.6, sharex=False, sharey=False):
#     fig, ax = plt.subplots(
#         nrows,
#         ncols,
#         figsize=(W, h * W),
#         sharex=sharex,
#         sharey=sharey,
#     )
#     for a in np.atleast_1d(ax).ravel():
#         a.tick_params(which="both", direction="in")
#     return fig, ax


# def finishfig(fig, config=None, OUT=None, name="figure", save=False, overwrite=False):
#     plt.tight_layout()

#     if save:
#         if OUT is None:
#             raise ValueError("OUT must be provided when save=True")

#         OUT = Path(OUT)
#         OUT.mkdir(parents=True, exist_ok=True)

#         plot_file = OUT / f"{name}.pdf"
#         pars_file = OUT / f"{name}.yaml"

#         if not overwrite and (plot_file.exists() or pars_file.exists()):
#             raise FileExistsError(f"{name} already exists in {OUT}")

#         meta = {"figure": plot_file.name, "simulation": config}
#         fig.savefig(plot_file, bbox_inches="tight")
#         pars_file.write_text(
#             yaml.safe_dump(meta, sort_keys=False),
#             encoding="utf-8",
#         )

#         print("Saved:", plot_file)
#         print("Saved:", pars_file)

#     plt.show()


# def addplot(run, fun, *args, name, **kwargs):
#     with TemporaryDirectory() as tmp:
#         fn = Path(tmp) / f"{name}.pdf"
#         fun(*args, OUT=tmp, name=name, save=True, overwrite=True, **kwargs)
#         run.add_artifact(str(fn), name=fn.name)


from pathlib import Path
from tempfile import TemporaryDirectory

import colormaps as cmaps
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm.auto import tqdm


FIG_W = 3.47412


class Plotter:
    """Hold figure defaults and the output target for one session or run."""

    def __init__(self, run=None, out=None, show=True, close=False, overwrite=False):
        self.run = run
        self.out = Path(out) if out is not None else None
        self.show = show
        self.close = close
        self.overwrite = overwrite

    def figure(
        self, name=None, shape=(1, 1), h=None, w=1.0,
        sharex=False, sharey=False, **kwargs,
    ):
        return FigureContext(
            self, name, shape, h, w, sharex, sharey, kwargs
        )

    def save(self, fig, name):
        if name is None:
            return

        if self.run is not None:
            with TemporaryDirectory() as tmp:
                fn = Path(tmp) / f"{name}.pdf"
                fig.savefig(fn, bbox_inches="tight")
                self.run.add_artifact(str(fn), name=fn.name)
            return

        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            fn = self.out / f"{name}.pdf"
            if fn.exists() and not self.overwrite:
                raise FileExistsError(f"{fn} already exists")
            fig.savefig(fn, bbox_inches="tight")
            print("Saved:", fn)


class FigureContext:
    """Create, finish, display, and optionally save one figure."""

    def __init__(self, pt, name, shape, h, w, sharex, sharey, kwargs):
        nr, nc = shape
        h = 0.6 * nr if h is None else h

        self.pt = pt
        self.name = name
        self.rect = None
        self._laid_out = False
        self.fig, self.ax = plt.subplots(
            nr, nc,
            figsize=(w * FIG_W, h * FIG_W),
            sharex=sharex,
            sharey=sharey,
            **kwargs,
        )

        for a in np.atleast_1d(self.ax).ravel():
            a.tick_params(which="both", direction="in")

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is not None:
            plt.close(self.fig)
            return False

        self.tight()
        self.pt.save(self.fig, self.name)

        if self.pt.show:
            plt.show()
        if self.pt.close:
            plt.close(self.fig)
        return False

    def layout(self, rect):
        self.rect = rect
        self._laid_out = False

    def tight(self):
        if not self._laid_out:
            self.fig.tight_layout(rect=self.rect)
            self._laid_out = True

    def colorbar(
        self, im, label, ax=None, cax=None, external=False,
        width="3%", height="100%",
    ):
        if external and cax is None:
            self.tight()
            a = np.atleast_1d(self.ax).ravel()[-1] if ax is None else ax
            cax = inset_axes(
                a, width=width, height=height, loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=a.transAxes, borderpad=0,
            )
            cax.set_in_layout(False)

        if cax is not None:
            cb = self.fig.colorbar(im, cax=cax)
        else:
            cb = self.fig.colorbar(im, ax=ax)

        cb.set_label(label, rotation=90, labelpad=5)
        cb.ax.tick_params(direction="in")
        return cb

    def sweepbar(
        self, sw, label, ax=None, cax=None, external=True,
        width="3%", height="100%",
    ):
        if external and cax is None:
            self.tight()
        a = self.ax if ax is None else ax
        return sw.colorbar(
            self.fig, a, label, cax=cax, external=external,
            width=width, height=height,
        )


class SweepColor:
    """Map one swept parameter to colors and a matching colorbar."""

    def __init__(self, par, cmap=cmaps.guppy, label=None):
        self.par = np.asarray(par)
        self.cmap = cmap
        self.label = label
        self.norm = Normalize(vmin=self.par.min(), vmax=self.par.max())
        self.sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.sm.set_array([])

    def __call__(self, val):
        return self.cmap(self.norm(val))

    def colorbar(
        self, fig, ax, label, cax=None, external=True,
        width="3%", height="100%",
    ):
        if external and cax is None:
            a = np.atleast_1d(ax).ravel()[-1]
            cax = inset_axes(
                a, width=width, height=height, loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=a.transAxes, borderpad=0,
            )
            cax.set_in_layout(False)

        if cax is not None:
            cb = fig.colorbar(self.sm, cax=cax)
        else:
            aa = np.atleast_1d(ax).ravel().tolist()
            cb = fig.colorbar(self.sm, ax=aa)

        cb.set_label(label, rotation=90, labelpad=5)
        cb.ax.tick_params(direction="in")
        return cb


def format_ax(
    ax, x=None, xlabel=None, ylabel=None, ylim=None,
    xticks=None, xticklabels=None, grid=True,
):
    """Apply the repeated axis formatting used by the project."""

    if x is not None:
        x = np.asarray(x)
        ax.set_xlim(x[0], x[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if grid:
        ax.grid(alpha=0.3)
    ax.tick_params(which="both", direction="in")
    return ax


def add_legend(ax, top=False, ncol=1, fontsize=8):
    if top:
        return ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=ncol, fontsize=fontsize,
        )
    return ax.legend(loc="best", ncol=ncol, fontsize=fontsize)


def add_fig_legend(fig, ax, ncol=1, top=0.92, fontsize=8):
    hh, ll = [], []
    for a in np.atleast_1d(ax).ravel():
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li and li not in ll:
                hh.append(hi)
                ll.append(li)

    fig.legend(
        hh, ll, bbox_to_anchor=(0, top, 1, 1 - top),
        loc="lower left", mode="expand", borderaxespad=0,
        ncol=ncol, fontsize=fontsize,
    )
    return 0, 0, 1, top - 0.01


def color_map(ax, x, y, z, cmap=cmaps.lipari, norm=None, vmin=None, vmax=None):
    im = ax.pcolormesh(
        x, y, z, shading="auto", cmap=cmap,
        norm=norm, vmin=vmin, vmax=vmax,
    )
    im.set_edgecolor("face")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    return im


def sweep_lines(ax, x, y, sw, labels=None, **kwargs):
    y = np.asarray(y)
    for i in tqdm(range(sw.par.size), desc="Plotting"):
        lb = None if labels is None else labels[i]
        ax.plot(x, y[i], color=sw(sw.par[i]), label=lb, **kwargs)
    return ax


def _plot_lines(ax, x, y, labels=None, sw=None, **kwargs):
    y = np.asarray(y)
    if isinstance(labels, str):
        labels = [labels]
    if y.ndim == 1:
        y = y[None, :]
    if y.ndim != 2:
        raise ValueError("y must have shape (nline, nx) or (nx,)")
    if sw is not None and sw.par.size != y.shape[0]:
        raise ValueError("the sweep and y must contain the same number of lines")
    if labels is not None and len(labels) != y.shape[0]:
        raise ValueError("labels and y must contain the same number of lines")

    kw = {"lw": 1.2, **kwargs}
    for i in tqdm(range(y.shape[0]), desc="Plotting"):
        cl = sw(sw.par[i]) if sw is not None else cm.Set2(i)
        lb = None if labels is None else labels[i]
        ax.plot(x, y[i], color=cl, label=lb, **kw)
    return ax


def plot_single(
    x, y, xlabel, ylabel, labels=None, sw=None,
    pt=None, name=None, ylim=None, **kwargs,
):
    """Plot one or more lines with either a legend or a sweep colorbar."""

    if labels is not None and sw is not None:
        raise ValueError("use labels or sw, not both")

    pt = Plotter() if pt is None else pt
    with pt.figure(name=name) as p:
        _plot_lines(p.ax, x, y, labels=labels, sw=sw, **kwargs)
        format_ax(p.ax, x=x, xlabel=xlabel, ylabel=ylabel, ylim=ylim)

        if labels is not None:
            add_legend(p.ax, ncol=min(len(labels), 4))
        elif sw is not None:
            p.sweepbar(sw, sw.label or r"$p$", ax=p.ax)

    return p.fig, p.ax


def plot_multi(
    x, ys, shape, xlabel, ylabels, labels=None, sw=None,
    pt=None, name=None, ylims=None, share=(False, False), **kwargs,
):
    """Plot an arbitrary panel grid with legends or one shared colorbar."""

    if labels is not None and sw is not None:
        raise ValueError("use labels or sw, not both")

    ys = list(ys)
    nr, nc = shape
    n = len(ys)
    if n > nr * nc:
        raise ValueError("shape does not contain enough axes")

    xa = np.asarray(x)
    xx = [xa] * n if xa.ndim == 1 else list(x)
    yl = [ylabels] * n if isinstance(ylabels, str) else list(ylabels)

    if labels is None:
        ll = [None] * n
    elif len(labels) > 0 and isinstance(labels[0], str):
        ll = [labels] * n
    else:
        ll = list(labels)

    if ylims is None or isinstance(ylims, tuple):
        ym = [ylims] * n
    else:
        ym = list(ylims)

    pt = Plotter() if pt is None else pt
    with pt.figure(
        name=name, shape=shape, h=max(0.6, 0.62 * nr),
        w=1.0 if nc == 1 else 2.1,
        sharex=share[0], sharey=share[1],
    ) as p:
        aa = np.atleast_1d(p.ax).ravel()

        for i in range(n):
            _plot_lines(aa[i], xx[i], ys[i], labels=ll[i], sw=sw, **kwargs)
            format_ax(
                aa[i], x=xx[i], xlabel=xlabel,
                ylabel=yl[i], ylim=ym[i],
            )
            if ll[i] is not None:
                add_legend(aa[i], ncol=min(len(ll[i]), 4))

        for a in aa[n:]:
            a.set_visible(False)

        if sw is not None:
            ch = f"{100 * nr + 8 * (nr - 1)}%"
            p.sweepbar(sw, sw.label or r"$p$", ax=aa[n - 1], height=ch)

    return p.fig, p.ax


def plot_cmap(
    x, y, z, label, cmap=cmaps.lipari, norm=None,
    pt=None, name=None, xlabel=r"$T_1/T_c$", ylabel=r"$T_2/T_c$",
):
    pt = Plotter() if pt is None else pt

    with pt.figure(name=name, h=0.85) as p:
        im = color_map(p.ax, x, y, z, cmap=cmap, norm=norm)
        p.colorbar(im, label, ax=p.ax)
        format_ax(p.ax, x=x, xlabel=xlabel, ylabel=ylabel, grid=False)

    return p.fig, p.ax


def plot_sweep_panels(
    x, ys, par, ylabels, pt=None, name=None,
    cmap=cmaps.guppy, xlabel=r"$T_1/T_c$", refs=None,
):
    pt = Plotter() if pt is None else pt
    ys = [np.asarray(y) for y in ys]
    n = len(ys)
    nc = 1 if n == 1 else 2
    nr = int(np.ceil(n / nc))
    sw = SweepColor(par, cmap)

    with pt.figure(
        name=name, shape=(nr, nc), h=max(0.6, 0.62 * nr),
        w=1.0 if nc == 1 else 2.1, sharex=True,
    ) as p:
        aa = np.atleast_1d(p.ax).ravel()

        for i, (a, y, yl) in enumerate(zip(aa, ys, ylabels)):
            sweep_lines(a, x, y, sw, lw=1.2)
            format_ax(a, x=x, xlabel=xlabel, ylabel=yl)
            if refs is not None and refs[i] is not None:
                a.axhline(refs[i], color="0.5", ls="--", lw=0.7)

        for a in aa[n:]:
            a.set_visible(False)

        ch = f"{100 * nr + 8 * (nr - 1)}%"
        p.sweepbar(sw, r"$V$", ax=aa[n - 1], height=ch)

    return p.fig, p.ax








#specific plots
def plot_phase_diag(da, d0, tn, pt=None, name="phase_diag"):
    return plot_cmap(
        tn, tn, da / d0, r"$\Delta/\Delta_0$",
        cmap=cmaps.lipari, pt=pt, name=name,
    )


def plot_nalpha_beta(occ, eq, par, path, x0, pt=None, name="nalpha_beta"):
    naa, nba = occ
    nea, neb = eq
    ks, kt, kl = path
    pt = Plotter() if pt is None else pt
    sw = SweepColor(par)

    with pt.figure(name=name, shape=(2, 1), h=1.15, sharex=True) as p:
        for q in tqdm(range(sw.par.size), desc="Plotting"):
            cl = sw(sw.par[q])
            p.ax[0].plot(ks, naa[q], color=cl, lw=1.3)
            p.ax[1].plot(ks, nba[q], color=cl, lw=1.3)

        lb = rf"eq, $T/T_c={x0:.1f}$"
        p.ax[0].plot(ks, nea, color="black", ls="--", lw=1.5, label=lb)
        p.ax[1].plot(ks, neb, color="black", ls="--", lw=1.5, label=lb)

        yl = [r"$n_{\alpha\mathbf{k}}$", r"$n_{\beta\mathbf{k}}$"]
        for a, y in zip(p.ax, yl):
            format_ax(a, x=ks, ylabel=y, xticks=kt, xticklabels=kl)
            add_legend(a, fontsize=7)
        p.ax[-1].set_xlabel(r"$\mathbf{k}$")
        p.sweepbar(sw, r"$T_1/T_2$", ax=p.ax[-1], height="211%")

    return p.fig, p.ax


def plot_nalpha_beta_bz(nfa, par, q0, pt=None, name="nalpha_beta_bz"):
    if q0 < 0 or q0 >= len(par):
        raise IndexError("q0 is outside the ratio sweep")

    pt = Plotter() if pt is None else pt
    nf = nfa[q0]
    ke = np.linspace(-1.0, 1.0, nf.shape[-1] + 1)

    with pt.figure(
        name=name, shape=(1, 2), h=0.62, sharex=True, sharey=True
    ) as p:
        im0 = p.ax[0].pcolormesh(
            ke, ke, nf[0].T, shading="flat",
            cmap=cmaps.lipari, vmin=0.0, vmax=1.0,
        )
        im1 = p.ax[1].pcolormesh(
            ke, ke, nf[1].T, shading="flat",
            cmap=cmaps.lipari, vmin=0.0, vmax=1.0,
        )
        im0.set_edgecolor("face")
        im1.set_edgecolor("face")

        p.ax[0].text(
            0.05, 0.93, r"$n_{\alpha\mathbf{k}}$",
            transform=p.ax[0].transAxes, color="white",
            ha="left", va="top",
        )
        p.ax[1].text(
            0.05, 0.93, r"$n_{\beta\mathbf{k}}$",
            transform=p.ax[1].transAxes, color="black",
            ha="left", va="top",
        )
        p.ax[1].text(
            0.95, 0.93, rf"$T_1/T_2={par[q0]:.2f}$",
            transform=p.ax[1].transAxes, color="black",
            ha="right", va="top",
        )

        for a in p.ax:
            format_ax(a, x=ke, xlabel=r"$k_x/\pi$", grid=False)
            a.set_ylim(ke[0], ke[-1])
            a.set_aspect("equal")
        p.ax[0].set_ylabel(r"$k_y/\pi$")
        p.colorbar(
            im1, r"$n_{\lambda\mathbf{k}}$", ax=p.ax[-1], external=True
        )

    return p.fig, p.ax


def plot_energies_uv(
    en, coh, har, par, path, ec=None, pt=None, name="energies_uv",
):
    eaa, eba = en
    ua, va = coh
    eah, ebh = har
    ks, kt, kl = path
    pt = Plotter() if pt is None else pt
    sw = SweepColor(par)

    with pt.figure(name=name, shape=(2, 1), h=1.15, sharex=True) as p:
        for q in tqdm(range(sw.par.size), desc="Plotting"):
            cl = sw(sw.par[q])
            p.ax[0].plot(ks, eaa[q], color=cl, ls="-", lw=1.2)
            p.ax[0].plot(ks, eba[q], color=cl, ls="--", lw=1.2)
            p.ax[1].plot(ks, ua[q], color=cl, ls="-", lw=1.2)
            p.ax[1].plot(ks, va[q], color=cl, ls="--", lw=1.2)

        p.ax[0].plot(ks, eah[0], color="black", lw=0.5)
        p.ax[0].plot(ks, ebh[0], color="black", lw=0.5)
        p.ax[0].plot([], [], color="black", ls="-", label=r"$E_{\alpha\mathbf{k}}$")
        p.ax[0].plot([], [], color="black", ls="--", label=r"$E_{\beta\mathbf{k}}$")
        p.ax[1].plot([], [], color="black", ls="-", label=r"$u_{\mathbf{k}}$")
        p.ax[1].plot([], [], color="black", ls="--", label=r"$v_{\mathbf{k}}$")

        if ec is not None:
            p.ax[0].axhline(ec, color="black", ls="-.", lw=0.5)

        yl = [r"$E_{\lambda\mathbf{k}}/|t_a|$", r"$u_{\mathbf{k}},v_{\mathbf{k}}$"]
        for a, y in zip(p.ax, yl):
            format_ax(a, x=ks, ylabel=y, xticks=kt, xticklabels=kl)
            add_legend(a, ncol=2)
        p.ax[-1].set_xlabel(r"$\mathbf{k}$")
        p.sweepbar(sw, r"$T_1/T_2$", ax=p.ax[-1], height="208%")

    return p.fig, p.ax
