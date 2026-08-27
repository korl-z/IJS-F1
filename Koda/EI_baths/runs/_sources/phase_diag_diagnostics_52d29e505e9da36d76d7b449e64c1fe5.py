import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import yaml
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("JAX x64:", jax.config.x64_enabled)
print("JAX dtype:", jnp.asarray(1.0).dtype)

#kozmetika
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri"],
        "font.size": 8,
        "mathtext.fontset": "cm",
    }
)

gnuplot_barve = cm.gnuplot2_r(np.linspace(0.3, 1, 10))
gnuplot_custom = LinearSegmentedColormap.from_list("gnuplot_custom", gnuplot_barve)

brg_barve = cmaps.guppy(np.linspace(0, 1, 10))
bgr_custom = LinearSegmentedColormap.from_list("bgr_custom", brg_barve)

YlGn_barve = cm.YlGn(np.linspace(0, 0.8, 10))
ylgn_custom = LinearSegmentedColormap.from_list("ylgn_custom", YlGn_barve)

markers = ["o", "D", "^", "x", "v", "<", ">", "p", "*", "X"]

set1_list = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#f781bf","#999999","#a65628",
]
bold_list = ["#7f3c8d","#11a579","#3969ac","#e73f74","#f2b701","#80ba5a","#e68310","#008695","#cf1c90","#f97b72","#a5aa99"]

from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats("png", dpi=300)

#racunanje
from EI import ei_unified as eu
from EI import ei_jax as ej
from utils.plotting_utils import plot_phase_diag, plot_energies_uv, plot_nalpha_beta, plot_nalpha_beta_bz
from utils.logger import logger, tqdm_bar

OUT = "/home/kzeleznikar/IJS-F1/Koda/EI_baths/plots/"

#setup config, bands and params
cfg_path = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/config/config_test.yaml")
run_path = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/runs/")

ex = Experiment("phase_diag_diagnostics")
ex.add_config(str(cfg_path))
ex.observers.append(FileStorageObserver(run_path))

@ex.main
def main(_config, _run):
    cfg = _config

    md = cfg["model"]
    eq = cfg["equilibrium"]
    sc = cfg["scan"]
    op = cfg["open_system"]

    nk = md["nk"]
    bp = md["band"]["pars"]

    bd = eu.tb_2d(nk["scan"], **bp)
    bd_ref = eu.tb_2d(nk["reference"], **bp)

    p = eu.MFPars(**md["mean_field"])

    #references Delta0, Tc
    s0 = eu.solve_eq(bd_ref, p, **eq["initial"], **eq["solve"])

    d0 = abs(s0.d)
    mu0 = s0.mu

    # za normalizacijo
    tc = eu.critical_temperature(bd_ref, p, **eq["critical"])

    logger.info(f"Delta_0 = {d0}")
    logger.info(f"T_c     = {tc}")
    logger.info(f"mu0     = {mu0}")

    #temperature grid
    tg = sc["temperature"]

    tn = np.linspace( tg["min_ratio"], tg["max_ratio"], tg["nt"])

    t1a = tn * tc
    t2a = tn * tc
    nt = tg["nt"]

    #reference bath potenciali, zelo situational
    mua = np.empty(nt)

    de = s0.d
    me = s0.m

    #bath params
    bc = op["bath"]

    rf = getattr(ej, bc["rate"])

    bpars = dict(bc["pars"])
    bpars["rate"] = rf

    Ec = md["mean_field"]["v"] * md["mean_field"]["n"] /2 + (md["band"]["pars"]["ta"] + md["band"]["pars"]["tb"]) / (2 * (md["band"]["pars"]["ta"] - md["band"]["pars"]["tb"])) * (md["mean_field"]["v"] * s0.m - md["band"]["pars"]["gap"])
    logger.info(Ec)

    bpars["mu"] = Ec

    #plot phase diagram
    da = np.full((nt, nt), np.nan)
    ma = np.full((nt, nt), np.nan)
    er = np.full((nt, nt), np.nan)
    it = np.zeros((nt, nt), dtype=int)
    ok = np.zeros((nt, nt), dtype=bool)


    de = s0.d
    me = s0.m
    ds = sc["continuation"]["d_floor_frac"] * d0

    for j in tqdm(range(nt), desc="T2"):
        t2 = t2a[j]

        se = eu.solve_eq(bd, p, t=t2, d=max(abs(de), 1.0e-4), m=me, **eq["solve"])

        de = se.d
        me = se.m

        n = se.n.copy()
        d = se.d
        m = se.m

        for i in tqdm(range(j, nt), desc="T1", leave=False):
            t1 = t1a[i]

            bs = (
                ej.gam_db(t=t1, name="bath 1", **bpars),
                ej.gam_db(t=t2, name="bath 2", **bpars),
            )

            so = ej.solve_open(bd, p, n, bs, d=d, m=m, **op["solve"])

            da[j, i] = abs(so.d)
            ma[j, i] = so.m
            er[j, i] = so.err
            it[j, i] = so.it
            ok[j, i] = so.ok

            n = so.n.copy()
            d = max(abs(so.d), ds)
            m = so.m

            da[i, j] = da[j, i]
            ma[i, j] = ma[j, i]
            er[i, j] = er[j, i]
            it[i, j] = it[j, i]
            ok[i, j] = ok[j, i]

    #make plot PD:
    plot_phase_diag(da, d0, tn, config = cfg, OUT=OUT, name="phase_diag_test", save=False, overwrite=True)

    #make reference points
    rs = cfg["ratio_sweep"]

    x0 = rs["x0"]
    r = np.arange(rs["r_min"], rs["r_max"], rs["r_step"]) ** 2
    x1a = x0 * np.sqrt(r)
    x2a = x0 / np.sqrt(r)

    t0 = x0 * tc

    #use dense band
    band = getattr(eu, md["band"]["name"])
    bd_dense = band(md["nk"]["reference"], **md["band"]["pars"])

    se = eu.solve_eq(bd_dense, p, t=t0, d=max(abs(s0.d), 1.0e-4), m=s0.m, **eq["solve"])

    nk1 = int(round(np.sqrt(se.n.shape[1])))

    c = nk1 // 2
    q = np.arange(c + 1)

    i1 = c - q
    j1 = np.full_like(q, c)

    i2 = np.zeros(c, dtype=int)
    j2 = c - np.arange(1, c + 1)

    i3 = np.arange(1, c + 1)
    j3 = np.arange(1, c + 1)

    ii = np.concatenate((i1, i2, i3))
    jj = np.concatenate((j1, j2, j3))

    s1 = np.linspace(0.0, np.pi, c + 1)
    s2 = np.linspace(np.pi, 2.0 * np.pi, c + 1)[1:]
    s3 = np.linspace(2.0 * np.pi, (2.0 + np.sqrt(2.0)) * np.pi, c + 1)[1:]

    ks = np.concatenate((s1, s2, s3))

    kt = [0.0, np.pi, 2.0 * np.pi, (2.0 + np.sqrt(2.0)) * np.pi]
    kl = [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"]


    def path(n):
        n = np.asarray(n)
        na = n[0].reshape(nk1, nk1)[ii, jj]
        nb = n[1].reshape(nk1, nk1)[ii, jj]
        return na, nb


    def path1(z):
        z = np.asarray(z).reshape(nk1, nk1)
        return z[ii, jj]


    nea, neb = path(se.n)

    gam_pars = dict(op["bath"]["pars"])
    gam_pars["rate"] = getattr(ej, op["bath"]["rate"])

    naa = np.empty((x2a.size, ks.size))
    nba = np.empty_like(naa)
    da1 = np.empty(x2a.size)
    era = np.empty(x2a.size)
    oka = np.empty(x2a.size, dtype=bool)

    eaa = np.empty_like(naa)
    eba = np.empty_like(naa)
    ua = np.empty_like(naa)
    va = np.empty_like(naa)

    eah = np.empty_like(naa)
    ebh = np.empty_like(naa)

    nfa = np.empty((r.size, 2, nk1, nk1))

    n = se.n.copy()
    d = se.d
    m = se.m

    bar = tqdm(range(r.size), desc="Ratio sweep")

    for q in bar:
        x1 = x1a[q]
        x2 = x2a[q]

        t1 = x1 * tc
        t2 = x2 * tc

        bs = (
            ej.gam_db(t1, name="bath 1", **gam_pars),
            ej.gam_db(t2, name="bath 2", **gam_pars),
        )

        sp = ej.solve_open(bd_dense, p, n, bs, d=d, m=m, **op["solve"])

        nfa[q] = np.asarray(sp.n).reshape(2, nk1, nk1)
        naa[q], nba[q] = path(sp.n)
        da1[q] = abs(sp.d)
        era[q] = sp.err
        oka[q] = sp.ok

        eaa[q], eba[q] = path(sp.st.e)
        ua[q] = path1(sp.st.u)
        va[q] = path1(sp.st.v)

        eah[q] = path1(sp.st.eah)
        ebh[q] = path1(sp.st.ebh)

        n = sp.n.copy()
        d = sp.d
        m = sp.m

        bar.set_postfix(r=f"{r[q]:.2f}", t1=f"{x1:.2f}", t2=f"{x2:.2f}", d=f"{abs(sp.d):.3e}", err=f"{sp.err:.2e}")

    logger.info(f"All converged: {np.all(oka)}")
    logger.info(f"Largest error: {np.max(era)}")

    #plot dispersion
    ra = r
    plot_energies_uv(eaa, eba, ua, va, eah, ebh, ra, ks, kt, kl, OUT=OUT, name="energies_uv_test", save=False, overwrite=True, Ec=Ec)

    #plot n alpha, beta
    plot_nalpha_beta(naa, nba, nea, neb, r, ks, kt, kl, OUT=OUT, name="nalpha_beta_test", save=False, overwrite=True)
    q0 = 4
    plot_nalpha_beta_bz(nfa, r, q0, cfg, OUT=OUT, name="nalpha_beta_bz_test", save=False, overwrite=True)


if __name__ == "__main__":
    ex.run_commandline()