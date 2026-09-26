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

import EI.ei_jax_2 as ej
import EI.ei_unified as eu
from EI.ei_utils import gap_info
import EI.ei_phonon as pb


CFG = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/config/config_test.yaml")
RUNS = Path("/home/kzeleznikar/IJS-F1/Koda/EI_baths/runs")
SHOW = False

# python phase_map_plots.py --id=phase_maps_v5_20260904 #za custom imena folderjev


with CFG.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["phase_maps"] = {"show": SHOW}
RUNS.mkdir(parents=True, exist_ok=True)

ex = Experiment("two_bath_phase_maps")
ex.add_config(cfg)
ex.observers.append(FileStorageObserver.create(str(RUNS)))


def make_baths(t1, t2, bp):
    """build baths"""
    b1 = ej.gam_db(t=t1, name="bath 1", **bp)
    b2 = ej.gam_db(t=t2, name="bath 2", **bp)
    return b1, b2


def put_sym(a, i, j, x):
    """simetrizita cmap"""
    a[j, i] = x
    a[i, j] = x


def scan_maps(bd, p, eq, op, ta, d0, m0, mu):
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

    bc = op["bath"]
    bp = {**bc["pars"], "rate": getattr(ej, bc["rate"]), "mu": mu}
    es = {**eq["solve"], "prog": False}
    de, me = d0, m0
    ds = 0.2 * max(d0, 1.0)

    for j in tqdm_bar(range(nt), desc="T2"):
        t2 = ta[j]
        se = eu.solve_eq(bd, p, t=t2, d=max(abs(de), 1.0e-8), m=me, **es)
        n, d, m = se.n.copy(), se.d, se.m

        for i in tqdm_bar(range(j, nt), desc="T1", leave=False):
            st = ej.solve_open(
                bd, p, n, make_baths(ta[i], t2, bp),
                d=d, m=m, **op["solve"],
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

            n, d, m = st.n.copy(), max(abs(st.d), ds), st.m

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
        ("imbalance_map", z["m"],
         r"$(n_a-n_b)$", cmaps.gem),
    )

    for name, val, label, cmap in tqdm_bar(maps, desc="Saving maps"):
        plot_cmap(tn, tn, val, label, cmap=cmap, pt=pt, name=name)


def save_data(run, tn, z, d0, tc):
    """store data"""
    with TemporaryDirectory() as tmp:
        fn = Path(tmp) / "phase_maps.npz"
        np.savez_compressed(fn, tn=tn, d0=d0, tc=tc, **z)
        run.add_artifact(str(fn), name=fn.name)


GAPS = {
    "hartree": "hartree_g",
    "diag": "diag_g",
    "indirect": "diag_ind",
    "diag_min": "diag_min",
    "hartree_ind": "hartree_ind",
}


def put_one(a, i, j, x):
    """Row = T2 index j, column = T1 index i (matches plot_cmap axes)."""
    a[j, i] = x


def make_ph_baths(t1, t2, k, bc, ov=({}, {})):
    """Two phonon baths: shared yaml pars, per-bath overrides in ov."""
    fn = getattr(pb, bc["rate"])
    return tuple(
        fn(t, k, name=f"bath_{i + 1}", **{**bc["pars"], **o})
        for i, (t, o) in enumerate(zip((t1, t2), ov))
    )


def scan_maps_ph(bd, p, eq, op, ta, d0, m0, sk, ov=({}, {}),
                 solve=ej.solve_fixed):
    """Two-bath phase maps for phonon baths, z[q][j, i] at T1=ta[i], T2=ta[j]."""
    nt = ta.size
    sh = (nt, nt)
    z = {q: np.full(sh, np.nan) for q in ("delta", "m", "error", *GAPS)}
    z["iterations"] = np.zeros(sh, dtype=int)
    z["converged"] = np.zeros(sh, dtype=bool)

    sym = ov[0] == ov[1]                   # identical baths: map is symmetric
    put = put_sym if sym else put_one
    k = np.asarray(bd.k, dtype=float)
    es = {**eq["solve"], "prog": False}
    ds = 0.2 * max(d0, 1.0)
    de, me = d0, m0

    for j in tqdm_bar(range(nt), desc="T2"):
        t2 = ta[j]
        se = eu.solve_eq(bd, p, t=t2, d=max(abs(de), 1.0e-8), m=me, **es)
        up = range(j, nt)
        dw = range(0) if sym else range(j - 1, -1, -1)

        for sw in (up, dw):
            n, d, m = se.n.copy(), se.d, se.m      # each sweep starts at T1 = T2
            for i in tqdm_bar(sw, desc="T1", leave=False):
                bs = make_ph_baths(ta[i], t2, k, op["bath"], ov)
                try:
                    st = solve(bd, p, n, bs, d=d, m=m, **sk)
                except RuntimeError as ex:
                    logger.warning("T1=%.4g, T2=%.4g failed: %s", ta[i], t2, ex)
                    n, d, m = se.n.copy(), max(abs(se.d), ds), se.m
                    continue

                gp = gap_info(bd, st.st)
                vs = {
                    "delta": abs(st.d), "m": st.m, "error": st.err,
                    "iterations": st.it, "converged": st.ok,
                    **{q: gp[g] for q, g in GAPS.items()},
                }
                for q, x in vs.items():
                    put(z[q], i, j, x)

                if st.ok:
                    n, d, m = st.n.copy(), max(abs(st.d), ds), st.m
                else:
                    n, d, m = se.n.copy(), max(abs(se.d), ds), se.m

        de, me = se.d, se.m

    return z


def scan_rect_ph(bd, p, eq, op, t1a, t2a, d0, m0, sk, ov=({}, {}),
                 solve=ej.solve_fixed):
    """Phonon maps on an independent grid, z[q][j, i] at T1=t1a[i], T2=t2a[j]."""
    n1, n2 = t1a.size, t2a.size
    sh = (n2, n1)
    z = {q: np.full(sh, np.nan) for q in ("delta", "m", "error", *GAPS)}
    z["iterations"] = np.zeros(sh, dtype=int)
    z["converged"] = np.zeros(sh, dtype=bool)

    k = np.asarray(bd.k, dtype=float)
    es = {**eq["solve"], "prog": False}
    ds = 0.2 * max(d0, 1.0)
    de, me = d0, m0

    for j in tqdm_bar(range(n2), desc="T2"):
        t2 = t2a[j]
        se = eu.solve_eq(bd, p, t=t2, d=max(abs(de), 1.0e-8), m=me, **es)
        i0 = int(np.argmin(np.abs(t1a - t2)))     # T1 closest to T2
        ex = bool(np.isclose(t1a[i0], t2))        # on the diagonal, eq is exact

        for sw in (range(i0, n1), range(i0 - 1, -1, -1)):
            n, d, m = se.n.copy(), se.d, se.m     # each sweep starts from eq(T2)
            for i in tqdm_bar(sw, desc="T1", leave=False):
                if not (ex and i == i0):
                    d = max(abs(d), ds)           # never seed off-diagonal at Delta = 0
                bs = make_ph_baths(t1a[i], t2, k, op["bath"], ov)
                try:
                    st = solve(bd, p, n, bs, d=d, m=m, **sk)
                except RuntimeError as er:
                    logger.warning("T1=%.4g, T2=%.4g failed: %s", t1a[i], t2, er)
                    n, d, m = se.n.copy(), se.d, se.m
                    continue

                gp = gap_info(bd, st.st)
                vs = {
                    "delta": abs(st.d), "m": st.m, "error": st.err,
                    "iterations": st.it, "converged": st.ok,
                    **{q: gp[g] for q, g in GAPS.items()},
                }
                for q, x in vs.items():
                    put_one(z[q], i, j, x)

                if st.ok:
                    n, d, m = st.n.copy(), st.d, st.m
                else:
                    n, d, m = se.n.copy(), se.d, se.m

        de, me = se.d, se.m

    return z


def plot_maps_rect(pt, r1, r2, z, d0):
    """plot_maps for independent T1, T2 axes; nonconverged points masked."""
    maps = (
        ("gap_map", z["delta"] / d0, r"$\Delta/\Delta_0$", cmaps.lipari),
        ("hartree_gap_map", z["hartree"] / d0,
         r"$G^{\mathrm{H}}(\Gamma)/\Delta_0$", cmaps.bubblegum),
        ("diagonal_gap_map", z["diag"] / d0,
         r"$G^{\mathrm{diag}}(\Gamma)/\Delta_0$", cmaps.batlow),
        ("min_direct_gap_map", z["diag_min"] / d0,
         r"$G^{\mathrm{diag}}_{\min}/\Delta_0$", cmaps.batlow),
        ("indirect_gap_map", z["indirect"] / (2.0 * d0),
         r"$G_{\mathrm{ind}}^{\mathrm{diag}}/(2\Delta_0)$", cmaps.amethyst),
        ("hartree_indirect_map", z["hartree_ind"] / d0,
         r"$G^{\mathrm{H}}_{\mathrm{ind}}/\Delta_0$", cmaps.bubblegum),
        ("imbalance_map", z["m"],
         r"$(n_a-n_b)$", cmaps.gem),
    )
    for name, val, label, cmap in tqdm_bar(maps, desc="Maps"):
        val = np.where(z["converged"], val, np.nan)
        plot_cmap(r1, r2, val, label, cmap=cmap, pt=pt, name=name)

        
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

    mf = model["mean_field"]
    v = float(mf["v"])
    n = float(mf["n"])
    ga = float(bp["gap"])
    aa = float(bp["ta"])
    ab = float(bp["tb"])
    mu = 0.5 * v * n + 0.5 * (aa + ab) / (aa - ab) * (v * s0.m - ga)

    logger.info("nk_scan = %d, nk_reference = %d", nk["scan"], nk["reference"])
    logger.info("Delta_0 = %.8f, m_0 = %.8f", d0, s0.m)
    logger.info("T_c = %.8f, bath center = %.8f", tc, mu)

    z = scan_maps(bd, p, equilibrium, open_system, ta, d0, s0.m, mu)
    nok = int(np.count_nonzero(~z["converged"]))
    emax = float(np.nanmax(z["error"]))
    logger.info("Nonconverged points = %d, maximum error = %.3e", nok, emax)

    _run.info["phase_maps"] = {
        "nk_scan": int(nk["scan"]),
        "nk_reference": int(nk["reference"]),
        "delta_0": d0,
        "m_0": float(s0.m),
        "tc": tc,
        "bath_center": float(mu),
        "nonconverged": nok,
        "max_error": emax,
    }
    _run.log_scalar("nonconverged", nok)
    _run.log_scalar("max_error", emax)

    pt = Plotter(run=_run, show=bool(phase_maps["show"]), close=True)
    plot_maps(pt, tn, z, d0)
    save_data(_run, tn, z, d0, tc)
    logger.info("Saved five maps and phase_maps.npz in Sacred run %s", _run._id)
