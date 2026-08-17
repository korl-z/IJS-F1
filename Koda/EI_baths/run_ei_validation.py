"""Run small function tests and validate ei_unified against utils.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ei_unified as eu


set1_list = list(plt.get_cmap("Set1").colors)


def pars(nk=9):
    return {
        "nk": nk,
        "gap": 2.0,
        "ta": 1.0,
        "tb": -0.3,
        "v": 2.5,
        "n0": 1.0,
        "t": 0.1,
    }


def test_bands(pa):
    print("\n1. Band constructors")
    b2 = eu.tb_2d(pa["nk"], pa["gap"], pa["ta"], pa["tb"])
    b1 = eu.tb_1d(11, pa["gap"], pa["ta"], pa["tb"])
    bf = eu.free_1d(11, 2.0, pa["gap"])
    q = np.linspace(-2.0, 2.0, 13)
    bc = eu.from_fn(q, lambda x: (0.5 * x * x, -0.3 * x * x))
    z = eu.reshape_mode(b2.ea, b2)

    assert b2.k.shape == (pa["nk"] ** 2, 2)
    assert z.shape == (pa["nk"], pa["nk"])
    assert np.isclose(np.sum(b2.w), 1.0)
    assert b1.dim == bf.dim == bc.dim == 1
    print("2D grid:", b2.shape, "modes:", b2.size)
    print("1D constructors:", b1.size, bf.size, bc.size)
    return b2


def test_mf(bd, pa):
    print("\n2. Mean field, FD state, and observables")
    p = eu.MFPars(pa["v"], pa["n0"])
    assert p.h == p.v
    assert eu.MFPars(pa["v"], pa["n0"], uh=0.0).h == 0.0

    st = eu.mf_state(bd, p, 0.3, -0.2)
    fs = eu.fermi(st, pa["t"], prog=True)
    f0 = eu.fermi_zero(st)
    na, nb = eu.band_occ(st, fs.n)
    tg = eu.targets(st, fs.n)

    assert st.e.shape == (2, bd.size)
    assert fs.n.shape == st.e.shape
    assert np.isclose(tg.n, pa["n0"], atol=1.0e-11)
    assert np.allclose(na + nb, np.sum(fs.n, axis=0))
    assert np.isclose(eu.targets(st, f0.n).n, pa["n0"])
    print("Default UH:", p.h)
    print("FD filling:", tg.n, "mu:", fs.mu)
    print("Targets:", tg.d, tg.m)
    return p, st, fs


def test_rates(st, fs, pa):
    print("\n3. Gamma functions and kinetic current")
    ga = eu.gam_db(pa["t"], kap=0.3, wc=5.0, orb="a")
    r = ga(st, fs.n)
    db = eu.check_db(r, st, pa["t"])
    dn = eu.dense_current(fs.n, r, st)
    nr = eu.number_rate(dn, st.bd)

    assert r.shape == (2, st.bd.size, 2, st.bd.size)
    assert np.all(np.isfinite(r)) and np.all(r >= 0.0)
    assert db < 1.0e-10
    assert abs(nr) < 1.0e-12

    rs = eu.gam_scalar(
        lambda a, k, b, q, s, n: 0.0 if (a == b and k == q) else 0.01
    )
    bt = eu.tb_1d(3, 1.0, 1.0, -0.3)
    pt = eu.MFPars(1.0)
    stt = eu.mf_state(bt, pt, 0.1, 0.0)
    nt = eu.fermi(stt, 0.2).n
    rt = rs(stt, nt)
    assert rt.shape == (2, 3, 2, 3)

    h = eu.lim_step(np.array([[0.9]]), np.array([[1.0]]), 1.0)
    assert 0.0 < h < 0.1
    print("Rate shape:", r.shape)
    print("Detailed-balance error:", db)
    print("Weighted number rate:", nr)


def test_eq(bd, p, pa):
    print("\n4. Equilibrium and critical-temperature solvers")
    sn = eu.solve_normal(bd, p, pa["t"], prog=True)
    la, _ = eu.pair_lambda(bd, p, pa["t"], m=sn.m)
    tc = eu.critical_temperature(
        bd,
        p,
        tlo=0.01,
        thi=2.0,
        tol=1.0e-6,
        nmax=50,
        prog=True,
    )
    se = eu.solve_eq(
        bd,
        p,
        pa["t"],
        d=0.4,
        m=0.0,
        mix=(0.2, 0.2),
        tol=1.0e-9,
        nmax=10000,
        chk=20,
        prog=True,
    )
    assert sn.ok and se.ok
    assert pa["t"] < tc
    assert la > 1.0 and se.d > 0.0
    print("Normal m:", sn.m)
    print("Pair eigenvalue:", la)
    print("Tc:", tc)
    print("Equilibrium Delta, m, mu:", se.d, se.m, se.mu)
    return se, tc


def test_open(bd, p, pa, se):
    print("\n5. Equal-temperature two-bath solver")
    di = 0.8 * se.d
    mi = np.clip(se.m + 0.03, -0.99, 0.99)
    si = eu.mf_state(bd, p, di, mi)
    ni = eu.fermi(si, 1.5 * pa["t"]).n

    d1 = eu.Dissipator(
        eu.gam_db(pa["t"], kap=0.3, wc=5.0),
        name="bath 1",
    )
    d2 = eu.Dissipator(
        eu.gam_db(pa["t"], kap=0.3, wc=5.0),
        name="bath 2",
    )
    dn = eu.total_current((d1, d2), si, ni)
    assert abs(eu.number_rate(dn, bd)) < 1.0e-12

    so = eu.solve_open(
        bd,
        p,
        ni,
        (d1, d2),
        d=di,
        m=mi,
        dt=0.8,
        td=0.3,
        tm=0.3,
        tol=1.0e-7,
        nmax=10000,
        chk=10,
        prog=True,
    )
    nf = eu.fermi(so.st, pa["t"]).n
    assert so.ok
    assert np.max(np.abs(so.n - nf)) < 2.0e-5
    print("Open Delta, m:", so.d, so.m)
    print("FD error:", np.max(np.abs(so.n - nf)))
    print("Open residual:", so.err)
    return so


def _stubs():
    if importlib.util.find_spec("tqdm") is None:
        z = types.ModuleType("tqdm")
        z.tqdm = lambda x, *a, **k: x
        sys.modules["tqdm"] = z
    if importlib.util.find_spec("colormaps") is None:
        sys.modules["colormaps"] = types.ModuleType("colormaps")
    if importlib.util.find_spec("labellines") is None:
        z = types.ModuleType("labellines")
        z.labelLines = lambda *a, **k: None
        sys.modules["labellines"] = z
    if importlib.util.find_spec("jax") is None:
        z = types.ModuleType("jax")
        z.random = types.ModuleType("jax.random")
        z.vmap = lambda f, *a, **k: f
        sys.modules["jax"] = z
        sys.modules["jax.numpy"] = np
        sys.modules["jax.random"] = z.random


def load_utils():
    _stubs()
    root = Path(__file__).resolve().parent
    cand = (root / "utils.py", root / "upload" / "utils.py")
    path = next((q for q in cand if q.exists()), None)
    if path is None:
        raise FileNotFoundError("Place utils.py beside this runner")
    sp = importlib.util.spec_from_file_location("utils_ref", path)
    md = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = md
    sp.loader.exec_module(md)
    return md


def test_utils(pa, se, so):
    print("\n6. Validation against utils.py")
    old = load_utils()
    bd = old.sq(pa["nk"], pa["gap"], pa["ta"], pa["tb"])
    p = old.Pars(v=pa["v"], t=pa["t"], n=pa["n0"], uh=None)
    sr = old.solve(bd, p, y0=(se.d, se.m, se.mu), tol=1.0e-10, nmax=3000)

    ed = abs(se.d - sr.d)
    em = abs(se.m - sr.m)
    eu0 = abs(se.mu - sr.mu)
    bd0 = abs(so.d - sr.d)
    bm0 = abs(so.m - sr.m)

    assert ed < 1.0e-7 and em < 1.0e-7 and eu0 < 1.0e-7
    assert bd0 < 5.0e-4 and bm0 < 5.0e-4
    print("                     utils        unified       two baths")
    print(f"Delta       {sr.d:14.10f} {se.d:14.10f} {so.d:14.10f}")
    print(f"m           {sr.m:14.10f} {se.m:14.10f} {so.m:14.10f}")
    print(f"Delta diff                 {ed:10.3e} {bd0:14.3e}")
    print(f"m diff                     {em:10.3e} {bm0:14.3e}")
    return sr


def _seg(a, b, nq):
    z = np.linspace(0.0, 1.0, nq, endpoint=False)
    return a[None, :] + z[:, None] * (b - a)[None, :]


def plot_disp(pa, se):
    nq = 100
    g = np.array([0.0, 0.0])
    x = np.array([np.pi, 0.0])
    m = np.array([np.pi, np.pi])
    k = np.vstack((_seg(g, x, nq), _seg(x, m, nq),
                   _seg(m, g, nq), g[None, :]))
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(k, axis=0), axis=1))))
    c = np.cos(k[:, 0]) + np.cos(k[:, 1])
    ea = 0.5 * pa["gap"] - 2.0 * pa["ta"] * c
    eb = -0.5 * pa["gap"] - 2.0 * pa["tb"] * c
    na = 0.5 * (pa["n0"] + se.m)
    nb = 0.5 * (pa["n0"] - se.m)
    eah = ea + pa["v"] * nb
    ebh = eb + pa["v"] * na
    av = 0.5 * (eah + ebh)
    xi = 0.5 * (eah - ebh)
    ek = np.sqrt(xi * xi + se.d * se.d)

    h = 0.8
    fig, ax = plt.subplots(figsize=(3.47412, h * 3.47412))
    ax.plot(s, eah, "--", color=set1_list[0], label=r"$\epsilon_a^{\rm H}$")
    ax.plot(s, ebh, "--", color=set1_list[1], label=r"$\epsilon_b^{\rm H}$")
    ax.plot(s, av + ek, color=set1_list[2], label=r"$E_\alpha$")
    ax.plot(s, av - ek, color=set1_list[3], label=r"$E_\beta$")
    ix = np.array([0, nq, 2 * nq, 3 * nq])
    ax.set_xticks(s[ix])
    ax.set_xticklabels([r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_xlim(s[0], s[-1])
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=2)
    plt.tight_layout()


def plot_conv(se, so):
    z = so.hist
    x = z["t"]
    h = 0.7
    fig, ax = plt.subplots(figsize=(3.47412, h * 3.47412))
    ax.plot(x, z["d"], color=set1_list[0], label=r"$\Delta_{\rm bath}$")
    ax.plot(x, np.full_like(x, se.d), "--", color=set1_list[1],
            label=r"$\Delta_{\rm eq}$")
    ax.plot(x, z["m"], color=set1_list[2], label=r"$m_{\rm bath}$")
    ax.plot(x, np.full_like(x, se.m), "--", color=set1_list[3],
            label=r"$m_{\rm eq}$")
    ax.set_xlabel(r"$t_{\rm rel}$")
    ax.set_ylabel(r"self-consistent fields")
    ax.set_xlim(x[0], x[-1])
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=2)
    plt.tight_layout()


def plot_fd(pa, so):
    nf = eu.fermi(so.st, pa["t"])
    x = np.linspace(so.st.e.min(), so.st.e.max(), 300)
    f = 1.0 / (np.exp(np.clip((x - nf.mu) / pa["t"], -700.0, 700.0)) + 1.0)
    h = 0.7
    fig, ax = plt.subplots(figsize=(3.47412, h * 3.47412))
    ax.scatter(so.st.e[0], so.n[0], s=8, color=set1_list[0], label=r"$n_\alpha$")
    ax.scatter(so.st.e[1], so.n[1], s=8, color=set1_list[1], label=r"$n_\beta$")
    ax.plot(x, f, color=set1_list[2], label=r"$f(E)$")
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"occupation")
    ax.set_xlim(x[0], x[-1])
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nk", type=int, default=9)
    ap.add_argument("--no-plots", action="store_true")
    ar = ap.parse_args()

    pa = pars(ar.nk)
    bd = test_bands(pa)
    p, st, fs = test_mf(bd, pa)
    test_rates(st, fs, pa)
    se, tc = test_eq(bd, p, pa)
    so = test_open(bd, p, pa, se)
    test_utils(pa, se, so)

    print("\nAll validation checks passed")
    print("Tc:", tc)

    if not ar.no_plots:
        plot_disp(pa, se)
        plot_conv(se, so)
        plot_fd(pa, so)
        plt.show()


if __name__ == "__main__":
    main()
