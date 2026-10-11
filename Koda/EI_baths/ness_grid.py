"""Solve one two-bath steady state on the fixed k grid (EI_grid). No plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from utils.logger import logger

import EI.ei_unified as eu
import EI_grid as eg

CFG = Path(__file__).resolve().parent / "config" / "config_grid.yaml"


def load(fn=CFG):
    with Path(fn).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup(cfg):
    """Scan bands, reference bands, mean field parameters and pair geometry."""
    md = cfg["model"]
    bp = md["band"]["pars"]
    bd = eu.tb_2d(md["nk"]["scan"], **bp)
    br = eu.tb_2d(md["nk"]["reference"], **bp)
    p = eu.MFPars(**md["mean_field"])
    geo = eg.make_geom(bd, bp["ta"], bp["tb"], **md["dipole"])
    return bd, br, p, geo


def make_baths(t1, t2, bc, ov=({}, {})):
    """Two baths: shared yaml pars, per-bath overrides in ov."""
    return tuple(
        eg.Bath(t, name=f"bath_{i + 1}", **{**bc["pars"], **o})
        for i, (t, o) in enumerate(zip((t1, t2), ov))
    )


def main(cfg=None):
    cfg = load() if cfg is None else cfg
    eq, op, sp = cfg["equilibrium"], cfg["open_system"], cfg["state"]
    bd, br, p, geo = setup(cfg)

    s0 = eu.solve_eq(br, p, **eq["initial"], **eq["solve"])
    d0 = max(abs(float(s0.d)), 1.0e-12)
    tc = float(eu.critical_temperature(br, p, **eq["critical"]))
    t1, t2 = sp["t1_tc"] * tc, sp["t2_tc"] * tc
    logger.info("backend = %s, nk = %d", eg.backend, cfg["model"]["nk"]["scan"])
    logger.info("Delta_0 = %.8f, m_0 = %.8f, T_c = %.8f", d0, s0.m, tc)

    bs = make_baths(t1, t2, op["bath"])
    st = eg.solve_ness(bd, p, geo, bs, d=d0, m=s0.m, eq=eq["solve"], **op["solve"])

    hs = st.hist
    if not st.ok:
        logger.warning("Newton did not reach the target: lam = %s", hs["lam"])
    logger.info("T1/T_c = %.4g, T2/T_c = %.4g", sp["t1_tc"], sp["t2_tc"])
    logger.info("Delta = %.8f, Delta/Delta_0 = %.6f, m = %.8f", st.d, abs(st.d) / d0, st.m)
    logger.info("err = %.3e, iterations = %d, path = %s", st.err, st.it,
                np.round(hs["lam"], 4).tolist())
    logger.info("filling = %.12f, max |dn/dt| = %.3e, kap = %.3e",
                np.sum(bd.w * st.n.sum(axis=0)), np.max(np.abs(st.cur)), float(hs["kap"]))
    if "lost" in hs:
        logger.info("order lost at path fraction %.4f", float(hs["lost"]))
    if "chi" in hs:
        logger.info("normal state, V chi_0 = %.6f", float(hs["chi"]))
    return st


if __name__ == "__main__":
    main()
