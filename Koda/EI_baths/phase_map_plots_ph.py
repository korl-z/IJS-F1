"""Compute the symmetric two-bath phase maps and store them with Sacred."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import colormaps as cmaps
import numpy as np
import yaml
from sacred import Experiment
from sacred.observers import FileStorageObserver

from utils.cosmetics import apply_plt_style
from utils.logger import logger, tqdm_bar
from utils.plotting_utils import Plotter, plot_cmap

import EI.ei_jax as ej
import EI.ei_unified as eu
from EI.ei_utils import gap_info


CFG = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/config/config_test.yaml")
RUNS = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/runs")
SHOW = False
PH = {
    "cs": 1.0,
    "ktf": 1.0,
    "eta": 0.1,
    "amp": 0.1,
    "ca": 1.0,
    "cb": 1.0,
    "qmax": np.pi,
}

# python phase_map_plots.py --id=phase_maps_v5_20260904 #za custom imena folderjev


with CFG.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PH["qmax"] = np.pi / float(cfg["model"]["band"]["pars"].get("a0", 1.0))
cfg["phase_maps"] = {"show": SHOW}
cfg["open_system"]["bath"] = {"kind": "longitudinal_acoustic", **PH}
RUNS.mkdir(parents=True, exist_ok=True)

ex = Experiment("two_bath_phase_maps")
ex.add_config(cfg)
ex.observers.append(FileStorageObserver.create(str(RUNS)))


def make_k(bd):
    """Use the exact reciprocal-space grid stored with the electron bands."""
    k = np.asarray(bd.k, dtype=float)
    if k.shape != (bd.size, 2):
        raise ValueError("the phonon model needs a two-dimensional band grid")
    return k


def make_baths(t1, t2, k, ph):
    """Build two independent longitudinal acoustic phonon baths."""
    b1 = ej.gam_ph(t=t1, k=k, name="bath 1", **ph)
    b2 = ej.gam_ph(t=t2, k=k, name="bath 2", **ph)
    return b1, b2


def put_sym(a, i, j, x):
    """simetrizita cmap"""
    a[j, i] = x
    a[i, j] = x


def scan_maps(bd, p, eq, op, ta, d0, m0, k, ph):
    """mirrora trikotnik"""
    nt = ta.size
    sh = (nt, nt)
    da = np.full(sh, np.nan)
    ma = np.full(sh, np.nan)
    gh = np.full(sh, np.nan)
    gg = np.full(sh, np.nan)
    gi = np.full(sh, np.nan)
    er = np.full(sh, np.nan)
    it = np.zeros(sh, dtype=int)
    ok = np.zeros(sh, dtype=bool)

    es = {**eq["solve"], "prog": False}
    de, me = d0, m0
    ds = 1.0e-6 * max(d0, 1.0)

    for j in tqdm_bar(range(nt), desc="T2"):
        t2 = ta[j]
        se = eu.solve_eq(bd, p, t=t2, d=max(abs(de), 1.0e-8), m=me, **es)
        n, d, m = se.n.copy(), se.d, se.m

        for i in tqdm_bar(range(j, nt), desc="T1", leave=False):
            st = ej.solve_open(
                bd, p, n, make_baths(ta[i], t2, k, ph),
                d=d, m=m, **{**op["solve"], "mode": "dense"},
            )
            gp = gap_info(bd, st.st)

            put_sym(da, i, j, abs(st.d))
            put_sym(ma, i, j, st.m)
            put_sym(gh, i, j, gp["hartree_g"])
            put_sym(gg, i, j, gp["diag_g"])
            put_sym(gi, i, j, gp["diag_ind"])
            put_sym(er, i, j, st.err)
            put_sym(it, i, j, st.it)
            put_sym(ok, i, j, st.ok)

            if st.ok:
                n, d, m = st.n.copy(), max(abs(st.d), ds), st.m
            else:
                n, d, m = se.n.copy(), max(abs(se.d), ds), se.m

        de, me = se.d, se.m

    return {
        "delta": da,
        "m": ma,
        "hartree": gh,
        "diag": gg,
        "indirect": gi,
        "error": er,
        "iterations": it,
        "converged": ok,
    }


def plot_maps(pt, tn, z, d0):
    """save cmaps"""
    maps = (
        ("gap_map", z["delta"] / d0, r"$\Delta/\Delta_0$", cmaps.lipari),
        ("hartree_gap_map", z["hartree"] / d0,
         r"$G^{\mathrm{H}}(\Gamma)/\Delta_0$", cmaps.bubblegum),
        ("diagonal_gap_map", z["diag"] / d0,
         r"$G^{\mathrm{diag}}(\Gamma)/\Delta_0$", cmaps.batlow),
        ("indirect_gap_map", z["indirect"] / (2.0 * d0),
         r"$G_{\mathrm{ind}}^{\mathrm{diag}}/(2\Delta_0)$", cmaps.amethyst),
        ("imbalance_map", z["m"] / (2.0 * d0),
         r"$(n_a-n_b)/(2\Delta_0)$", cmaps.gem),
    )

    for name, val, label, cmap in tqdm_bar(maps, desc="Saving maps"):
        val = np.where(z["converged"], val, np.nan)
        plot_cmap(tn, tn, val, label, cmap=cmap, pt=pt, name=name)


def save_data(run, tn, z, d0, tc):
    """store data"""
    with TemporaryDirectory() as tmp:
        fn = Path(tmp) / "phase_maps.npz"
        np.savez_compressed(fn, tn=tn, d0=d0, tc=tc, **z)
        run.add_artifact(str(fn), name=fn.name)


@ex.automain
def main(_run, model, equilibrium, scan, open_system, phase_maps):
    """map loop run"""
    apply_plt_style()
    _run.add_resource(str(CFG))

    nk = model["nk"]
    bp = model["band"]["pars"]
    bd = eu.tb_2d(nk["scan"], **bp)
    br = eu.tb_2d(nk["reference"], **bp)
    p = eu.MFPars(**model["mean_field"])

    s0 = eu.solve_eq(br, p, **equilibrium["initial"], **equilibrium["solve"])
    d0 = max(abs(float(s0.d)), 1.0e-12)
    tc = float(eu.critical_temperature(br, p, **equilibrium["critical"]))

    tg = scan["temperature"]
    tn = np.linspace(tg["min_ratio"], tg["max_ratio"], tg["nt"])
    ta = tn * tc

    k = make_k(bd)
    ph = {key: open_system["bath"][key] for key in PH}

    logger.info("nk_scan = %d, nk_reference = %d", nk["scan"], nk["reference"])
    logger.info("Delta_0 = %.8f, m_0 = %.8f", d0, s0.m)
    logger.info("T_c = %.8f, phonon pars = %s", tc, ph)

    z = scan_maps(bd, p, equilibrium, open_system, ta, d0, s0.m, k, ph)
    nok = int(np.count_nonzero(~z["converged"]))
    emax = float(np.nanmax(z["error"]))
    logger.info("Nonconverged points = %d, maximum error = %.3e", nok, emax)

    _run.info["phase_maps"] = {
        "nk_scan": int(nk["scan"]),
        "nk_reference": int(nk["reference"]),
        "delta_0": d0,
        "m_0": float(s0.m),
        "tc": tc,
        "phonon": dict(ph),
        "nonconverged": nok,
        "max_error": emax,
    }
    _run.log_scalar("nonconverged", nok)
    _run.log_scalar("max_error", emax)

    pt = Plotter(run=_run, show=bool(phase_maps["show"]), close=True)
    plot_maps(pt, tn, z, d0)
    save_data(_run, tn, z, d0, tc)
    logger.info("Saved five maps and phase_maps.npz in Sacred run %s", _run._id)

    import EI.ei_phonon as ep
    
    k = np.asarray(bd.k)
    ph = dict(cs=0.2, ktf=1.0, eta=0.04, amp=0.1, qmax=np.pi)
    st = ej.mf_state(bd, p, d=float(s0.d), m=float(s0.m))
    
    b1 = ep.gam_ph(t=0.2 * tc, k=k, **ph)
    b2 = ep.gam_ph(t=3.0 * tc, k=k, **ph)
    
    r1 = np.asarray(ej.dense_rates(st, (b1,)))
    r2 = np.asarray(ej.dense_rates(st, (b2,)))
    r  = np.asarray(ej.dense_rates(st, (b1, b2)))
    
    nk = bd.size
    rin = r[:nk, :nk].sum() + r[nk:, nk:].sum()
    rcr = r[:nk, nk:].sum() + r[nk:, :nk].sum()
    
    q = k[:, None, :] - k[None, :, :]
    om = ph["cs"] * np.linalg.norm(q, axis=-1)
    valid = (om > 0) & np.all(np.abs(q) <= ph["qmax"], axis=-1)
    de = np.asarray(st.e)[0, :, None] - np.asarray(st.e)[1, None, :]
    det = np.minimum(np.abs(de - om), np.abs(de + om))
    
    print("Rate-sum error:", np.max(np.abs(r - r1 - r2)))
    print("Cold/hot rate difference:", np.max(np.abs(r2 - r1)))
    print("Interband/intraband rate:", rcr / max(rin, 1e-30))
    print("Largest allowed phonon energy:", np.max(om[valid]))
    print("Smallest interband detuning / eta:", np.min(det[valid]) / ph["eta"])