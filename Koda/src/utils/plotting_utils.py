import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import matplotlib.cm as cm
import colormaps as cmaps
import matplotlib.colors as colors
from labellines import labelLines
from matplotlib.cm import brg, ScalarMappable
from matplotlib.colors import (
    LogNorm,
    TwoSlopeNorm,
    LinearSegmentedColormap,
    Normalize,
    ListedColormap,
)
from matplotlib.lines import Line2D

import jax
import jax.numpy as jnp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import yaml
from pathlib import Path


def plot_phase_diag(
    da, d0, tn, config=None, OUT=None, name="phase_diag", save=False, overwrite=False
):
    z = da / d0

    h = 0.85
    cm = cmaps.lipari

    fig, ax = plt.subplots(figsize=(3.47412, h * 3.47412))

    im = ax.pcolormesh(tn, tn, z, shading="auto", cmap=cm, vmin=0.0)
    im.set_edgecolor("face")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta/\Delta_0$", rotation=90, labelpad=5)

    # probe points
    r = np.array(
        [
            x**2
            for x in np.arange(
                config["ratio_sweep"]["r_min"],
                config["ratio_sweep"]["r_max"],
                config["ratio_sweep"]["r_step"],
            )
        ]
    )
    ax.scatter(
        config["ratio_sweep"]["x0"] * np.sqrt(r),
        config["ratio_sweep"]["x0"] / np.sqrt(r),
        color="green",
    )

    ax.set_xlabel(r"$T_1/T_c$")
    ax.set_ylabel(r"$T_2/T_c$")
    ax.set_xlim(tn.min(), tn.max())
    ax.set_ylim(tn.min(), tn.max())

    plt.tight_layout()

    if save:
        if OUT is None:
            raise ValueError("OUT must be provided when save=True")

        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)
        
        if not overwrite and (plot_file.exists()):
            raise FileExistsError(f"{name} already exists in {OUT}")
        
        plot_file = OUT / f"{name}.pdf"
        fig.savefig(plot_file, bbox_inches="tight")

        print("Saved:", plot_file)

        plt.show()
        plt.close(fig)


def plot_nalpha_beta(
    naa,
    nba,
    nea,
    neb,
    r,
    ks,
    kt,
    kl,
    config,
    OUT=None,
    name="nalpha_beta",
    save=False,
    overwrite=False,
):
    x0 = config["ratio_sweep"]["x0"]
    ra = r

    norm = Normalize(vmin=ra.min(), vmax=ra.max())
    cm = cmaps.guppy
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    h = 1.15
    fig, ax = plt.subplots(2, 1, figsize=(3.47412, h * 3.47412), sharex=True)

    for q in tqdm(range(r.size), desc="Plotting"):
        cl = cm(norm(ra[q]))

        ax[0].plot(ks, naa[q], color=cl, linewidth=1.3)
        ax[1].plot(ks, nba[q], color=cl, linewidth=1.3)

    ax[0].plot(
        ks,
        nea,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"eq, $T/T_c={x0:.1f}$",
    )
    ax[1].plot(
        ks,
        neb,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"eq, $T/T_c={x0:.1f}$",
    )

    ax[0].set_ylabel(r"$n_{\alpha\mathbf{k}}$")
    ax[1].set_ylabel(r"$n_{\beta\mathbf{k}}$")
    ax[1].set_xlabel(r"$\mathbf{k}$")

    for a in ax:
        a.set_xlim(ks[0], ks[-1])
        a.set_xticks(kt)
        a.set_xticklabels(kl)
        a.grid(alpha=0.3)
        a.legend(loc="best", fontsize=7)

    cax = inset_axes(
        ax[1],
        width="3%",
        height="211%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0, 1, 1),
        bbox_transform=ax[1].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$T_1/T_2$", rotation=90, labelpad=5)

    plt.tight_layout()

    if save:
        if OUT is None:
            raise ValueError("OUT must be provided when save=True")

        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)

        plot_file = OUT / f"{name}.pdf"

        if not overwrite and (plot_file.exists()):
            raise FileExistsError(f"{name} already exists in {OUT}")

        fig.savefig(plot_file, bbox_inches="tight")

        print("Saved:", plot_file)

    plt.show()
    plt.close(fig)


def plot_nalpha_beta_bz(
    nfa, r, q0, OUT=None, name="nalpha_beta_bz", save=False, overwrite=False
):
    if q0 < 0 or q0 >= r.size:
        raise IndexError("q0 is outside the ratio sweep")

    nf = nfa[q0]
    nk1 = nf.shape[-1]

    ke = np.linspace(-1.0, 1.0, nk1 + 1)

    h = 0.62
    fig, ax = plt.subplots(
        1, 2, figsize=(3.47412, h * 3.47412), sharex=True, sharey=True
    )

    im0 = ax[0].pcolormesh(
        ke, ke, nf[0].T, shading="flat", cmap=cmaps.lipari, vmin=0.0, vmax=1.0
    )
    im0.set_edgecolor("face")

    im1 = ax[1].pcolormesh(
        ke, ke, nf[1].T, shading="flat", cmap=cmaps.lipari, vmin=0.0, vmax=1.0
    )
    im1.set_edgecolor("face")

    ax[0].text(
        0.05,
        0.93,
        r"$n_{\alpha\mathbf{k}}$",
        transform=ax[0].transAxes,
        color="white",
        ha="left",
        va="top",
    )

    ax[1].text(
        0.05,
        0.93,
        r"$n_{\beta\mathbf{k}}$",
        transform=ax[1].transAxes,
        color="k",
        ha="left",
        va="top",
    )

    ax[1].text(
        0.95,
        0.93,
        rf"$T_1/T_2={r[q0]:.2f}$",
        transform=ax[1].transAxes,
        color="k",
        ha="right",
        va="top",
    )
    ax[0].tick_params(which="both", direction="in")
    ax[1].tick_params(which="both", direction="in")

    for a in ax:
        a.set_xlim(ke[0], ke[-1])
        a.set_ylim(ke[0], ke[-1])
        a.set_aspect("equal")
        a.set_xlabel(r"$k_x/\pi$")

    ax[0].set_ylabel(r"$k_y/\pi$")

    cax = inset_axes(
        ax[1],
        width="3%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0, 1, 1),
        bbox_transform=ax[1].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label(r"$n_{\lambda\mathbf{k}}$", rotation=90, labelpad=5)

    plt.tight_layout()

    if save:
        if OUT is None:
            raise ValueError("OUT must be provided when save=True")

        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)

        plot_file = OUT / f"{name}.pdf"

        if not overwrite and (plot_file.exists()):
            raise FileExistsError(f"{name} already exists in {OUT}")

        fig.savefig(plot_file, bbox_inches="tight")

        print("Saved:", plot_file)

    plt.show()
    plt.close(fig)


def plot_energies_uv(
    eaa,
    eba,
    ua,
    va,
    eah,
    ebh,
    ra,
    ks,
    kt,
    kl,
    OUT=None,
    name="energies_uv",
    save=False,
    overwrite=False,
    Ec=None,
):
    norm = Normalize(vmin=ra.min(), vmax=ra.max())
    cm = cmaps.guppy
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    h = 1.15
    fig, ax = plt.subplots(2, 1, figsize=(3.47412, h * 3.47412), sharex=True)

    for q in tqdm(range(ra.size), desc="Plotting"):
        cl = cm(norm(ra[q]))

        ax[0].plot(ks, eaa[q], color=cl, linestyle="-", linewidth=1.2)
        ax[0].plot(ks, eba[q], color=cl, linestyle="--", linewidth=1.2)

        ax[1].plot(ks, ua[q], color=cl, linestyle="-", linewidth=1.2)
        ax[1].plot(ks, va[q], color=cl, linestyle="--", linewidth=1.2)

    ax[0].plot(ks, eah[0], color="black", linestyle="-", linewidth=0.5)
    ax[0].plot(ks, ebh[0], color="black", linestyle="-", linewidth=0.5)

    ax[0].plot([], [], color="black", linestyle="-", label=r"$E_{\alpha\mathbf{k}}$")
    ax[0].plot([], [], color="black", linestyle="--", label=r"$E_{\beta\mathbf{k}}$")

    ax[1].plot([], [], color="black", linestyle="-", label=r"$u_{\mathbf{k}}$")
    ax[1].plot([], [], color="black", linestyle="--", label=r"$v_{\mathbf{k}}$")

    ax[0].set_ylabel(r"$E_{\lambda\mathbf{k}}/|t_a|$")
    ax[1].set_ylabel(r"$u_{\mathbf{k}},\,v_{\mathbf{k}}$")
    ax[1].set_xlabel(r"$\mathbf{k}$")

    ax[0].tick_params(which="both", direction="in")
    ax[1].tick_params(which="both", direction="in")

    if Ec is not None:
        ax[0].axhline(
            Ec,
            xmin=0.0,
            xmax=1.0,
            color="k",
            linestyle="-.",
            linewidth=0.5,
        )

    for a in ax:
        a.set_xlim(ks[0], ks[-1])
        a.set_xticks(kt)
        a.set_xticklabels(kl)
        a.grid(alpha=0.3)
        a.legend(loc="best", ncol=2, fontsize=8)

    cax = inset_axes(
        ax[1],
        width="3%",
        height="208%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0, 1, 1),
        bbox_transform=ax[1].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$T_1/T_2$", rotation=90, labelpad=5)

    plt.tight_layout()

    if save:
        if OUT is None:
            raise ValueError("OUT must be provided when save=True")

        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)

        plot_file = OUT / f"{name}.pdf"

        if not overwrite and (plot_file.exists()):
            raise FileExistsError(f"{name} already exists in {OUT}")

        fig.savefig(plot_file, bbox_inches="tight")

        print("Saved:", plot_file)

    plt.show()
    plt.close(fig)


def newfig(nrows=1, ncols=1, W=3.47412, h=0.6, sharex=False, sharey=False):
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(W, h * W),
        sharex=sharex,
        sharey=sharey,
    )
    for a in np.atleast_1d(ax).ravel():
        a.tick_params(which="both", direction="in")
    return fig, ax


def finishfig(fig, config=None, OUT=None, name="figure", save=False, overwrite=False):
    plt.tight_layout()

    if save:
        if OUT is None:
            raise ValueError("OUT must be provided when save=True")

        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)

        plot_file = OUT / f"{name}.pdf"
        pars_file = OUT / f"{name}.yaml"

        if not overwrite and (plot_file.exists() or pars_file.exists()):
            raise FileExistsError(f"{name} already exists in {OUT}")

        meta = {"figure": plot_file.name, "simulation": config}
        fig.savefig(plot_file, bbox_inches="tight")
        pars_file.write_text(
            yaml.safe_dump(meta, sort_keys=False),
            encoding="utf-8",
        )

        print("Saved:", plot_file)
        print("Saved:", pars_file)

    plt.show()
