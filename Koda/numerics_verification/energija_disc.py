import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

# Koda/src na pot, ce skripta lezi v Koda/<mapa>/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import EI.ei_phonon as pb
import EI.ei_unified as eu
from utils.cosmetics import SET1_LIST, apply_plt_style
from utils.plotting_utils import Plotter, format_ax, add_fig_legend

# parametri kot v config_phonons.yaml
BAND = dict(gap=2.0, ta=1.0, tb=-1.0)
MF = dict(v=4.4, n=1.0, uh=None)
BATH = dict(amp=1.0, dip=2.0, cs=5.0, w0=2.2, gam=0.6,
            qd=4.442882938158366, nsub=16)


def cut_data(st, bd, b, mu, nu, ip, nf=41):
    """kx rez skozi p0: pasovi, Omega, tockovna in povprecena crta"""
    nx, ny = bd.shape
    kg = np.asarray(bd.k).reshape(nx, ny, 2)
    ix, iy = np.unravel_index(ip, (nx, ny))
    kx = kg[:, iy, 0]
    dk = 2.0 * np.pi / nx

    # Hartree pasova vzdolz reza: konst. - 2 t cos kx
    ea = np.asarray(st.eah).reshape(nx, ny)[:, iy]
    eb = np.asarray(st.ebh).reshape(nx, ny)[:, iy]
    ca = np.mean(ea + 2.0 * b.ta * np.cos(kx))
    cb = np.mean(eb + 2.0 * b.tb * np.cos(kx))

    # efektivni d iz razcepa na mrezi
    eg = np.asarray(st.e).reshape(2, nx, ny)[:, :, iy]
    xi = 0.5 * (ea - eb)
    dd = np.sqrt(max(np.mean((0.5 * (eg[1] - eg[0]))**2 - xi**2), 0.0))

    # fina mreza: nf tock na celico
    sf = np.linspace(-0.5, 0.5, nf)
    kf = kx[:, None] + dk * sf[None, :]
    fa = ca - 2.0 * b.ta * np.cos(kf)
    fb = cb - 2.0 * b.tb * np.cos(kf)
    av, xf = 0.5 * (fa + fb), 0.5 * (fa - fb)
    rf = np.sqrt(xf**2 + dd**2)
    br = np.stack((av - rf, av + rf))

    # veja, ki se ujema z E_nu na mrezi
    ec = br[:, :, nf // 2]
    jb = int(np.argmin(np.abs(ec - eg[nu]).max(axis=1)))
    e0 = eg[mu, ix]
    of = br[jb] - e0              # fina Omega, oblika (nx, nf)
    om = eg[nu] - e0              # Omega na mrezi
    err = np.abs(ec[jb] - eg[nu]).max()

    # prenos q = k - p0, zavit v BZ
    q0 = np.mod(kx - kx[ix] + np.pi, 2.0 * np.pi) - np.pi

    def line(w, q):
        wq = np.hypot(b.w0, b.cs * np.abs(q))
        s = np.asarray(pb._spec_e(w, wq, b.gam))
        return s * (np.abs(q) <= b.qd)

    sp = line(om, q0)              # tockovna
    sc = line(of, q0[:, None] + dk * sf[None, :])  # vzdolz pasu

    # 2D sotor, ista vozlisca kot v _ltab
    ns = max(int(b.nsub), 1)
    t = 2.0 * ((np.arange(ns) + 0.5) / ns - 0.5)
    wt = 1.0 - np.abs(t)
    wt = wt / wt.sum()
    sa = np.zeros_like(om)
    for tx, ax in zip(t, wt):
        for ty, ay in zip(t, wt):
            qx = q0 + tx * dk
            qy = ty * dk
            qq = np.hypot(qx, qy)
            wq = np.hypot(b.w0, b.cs * qq)
            s = np.asarray(pb._spec_e(om, wq, b.gam))
            sa += ax * ay * s * (qq <= b.qd)

    return dict(kx=kx, kf=kf, of=of, om=om, sp=sp, sc=sc, sa=sa,
                dk=dk, ix=ix, err=err)


def plot_cut(pt, cd, mu, nu, bl=(r"\alpha", r"\beta"), name=None):
    """E(k) z energijskimi bini celic in A(Omega) - A(-Omega) ob strani"""
    kx, kf, of, om = cd["kx"], cd["kf"], cd["of"], cd["om"]
    lo, hi = of.min(axis=1), of.max(axis=1)
    ke = np.r_[kx - 0.5 * cd["dk"], kx[-1] + 0.5 * cd["dk"]] / np.pi
    ym = (of.min(), of.max())
    dy = 0.05 * (ym[1] - ym[0])
    c0, c1, c2 = SET1_LIST[1], SET1_LIST[0], SET1_LIST[2]

    with pt.figure(name=name, shape=(1, 2), h=0.7, sharey=True,
                   gridspec_kw={"width_ratios": (1.6, 1.0)}) as q:
        a = q.ax[0]
        # energijski bin vsake celice
        for i in range(kx.size):
            a.add_patch(Rectangle((ke[i], lo[i]), ke[i + 1] - ke[i],
                                  hi[i] - lo[i], color="0.85", lw=0,
                                  zorder=0))
        for e in ke:
            a.axvline(e, color="0.75", lw=0.3, zorder=1)
        a.plot(kf.ravel() / np.pi, of.ravel(), color="0.3", lw=0.9,
               label=rf"$E_{{{bl[nu]} k}}-E_{{{bl[mu]} p_0}}$")
        a.scatter(kx / np.pi, om, color=c0, s=7, zorder=4, label=r"$k_i$")
        a.scatter(np.full_like(om, ke[0]), om, color=c0, s=5,
                  clip_on=False, zorder=5)
        a.axvline(kx[cd["ix"]] / np.pi, color="k", ls="--", lw=0.6,
                  label=r"$p_0$")
        format_ax(a, x=ke, xlabel=r"$k_x/\pi$", ylabel=r"$\Omega$",
                  ylim=(ym[0] - dy, ym[1] + dy), grid=False)

        a = q.ax[1]
        a.axvline(0.0, color="0.6", lw=0.5)
        a.plot(cd["sc"].ravel(), of.ravel(), color="k", lw=0.6,
               label="pas")
        a.scatter(cd["sp"], om, color=c1, s=7, zorder=4,
                  label="po tockah")
        a.errorbar(cd["sa"], om, yerr=(om - lo, hi - om), fmt="s",
                   color=c2, ms=2.5, lw=0.6, capsize=1.0, zorder=3,
                   label="povpreceno")
        sx = np.r_[cd["sc"].ravel(), cd["sp"], cd["sa"], 0.0]
        format_ax(a, xlabel=r"$A(\Omega)-A(-\Omega)$", grid=False)
        a.set_xlim(sx.min() * 1.05, sx.max() * 1.05)
        q.layout(add_fig_legend(q.fig, q.ax, ncol=3, top=0.84, fontsize=6))
    return q.fig, q.ax


def lut_data(st, bd, b, mu, nu, iq, wl=None):
    """ena vrstica tabele T(q_n, w) in vsi pari s prenosom q_n"""
    nx, ny = bd.shape
    dk = 2.0 * np.pi / nx
    qn = (np.mod(np.asarray(iq) * dk + np.pi, 2.0 * np.pi) - np.pi)

    # energije vseh parov p -> p + q_n
    eg = np.asarray(st.e).reshape(2, nx, ny)
    ek = np.roll(eg[nu], shift=(-iq[0], -iq[1]), axis=(0, 1))
    wp = (ek - eg[mu]).ravel()

    # sotor, ista vozlisca kot v _ltab
    ns = max(int(b.nsub), 1)
    t = 2.0 * ((np.arange(ns) + 0.5) / ns - 0.5)
    wt = 1.0 - np.abs(t)
    wt = wt / wt.sum()
    tx, ty = np.meshgrid(t, t, indexing="ij")
    aw = np.outer(wt, wt).ravel()
    qq = np.hypot(qn[0] + tx.ravel() * dk, qn[1] + ty.ravel() * dk)
    om = np.hypot(b.w0, b.cs * qq)
    aw = aw * (qq <= b.qd)
    oc = np.hypot(b.w0, b.cs * np.hypot(*qn))

    def avg(w):
        """enacba (20) pri poljubnem w, liha v w"""
        w = np.atleast_1d(w)
        s = np.asarray(pb._spec_e(np.abs(w)[:, None], om[None, :], b.gam))
        return np.sign(w) * (s @ aw)

    # tabela: dw = gam / 8, w_l >= 0
    dw = b.gam / 8.0
    wm = np.abs(wp).max() + 10.0 * b.gam if wl is None else wl
    nw = int(np.ceil(wm / dw)) + 2
    wg = dw * np.arange(nw)
    tg = avg(wg)

    # lookup, enacba (23), na |w|
    z = np.clip(np.abs(wp) / dw, 0.0, nw - 1)
    il = np.minimum(np.floor(z).astype(int), nw - 2)
    f = z - il
    tp = np.sign(wp) * ((1.0 - f) * tg[il] + f * tg[il + 1])

    # fina os za primerjavo
    wf = np.linspace(wp.min() - 5 * b.gam, wp.max() + 5 * b.gam, 4000)
    sp = np.asarray(pb._spec_e(np.abs(wf), oc, b.gam)) * np.sign(wf)
    sp = sp * (np.hypot(*qn) <= b.qd)
    sn = np.sign(wf)[:, None] * np.asarray(
        pb._spec_e(np.abs(wf)[:, None], om[None, :], b.gam))

    return dict(qn=qn, wp=wp, tp=tp, ex=avg(wp), wg=wg, tg=tg, dw=dw,
                wf=wf, sp=sp, tf=avg(wf), sn=sn, aw=aw, il=il, f=f)


def plot_lut(pt, ld, name=None, jz=None):
    """tabela v w, lookup parov in povecava enega lookupa"""
    wp, tp, wf = ld["wp"], ld["tp"], ld["wf"]
    wg, tg, dw = ld["wg"], ld["tg"], ld["dw"]
    jz = int(np.argmax(np.abs(tp))) if jz is None else jz
    c0, c1, c2 = SET1_LIST[1], SET1_LIST[0], SET1_LIST[2]

    with pt.figure(name=name, shape=(1, 2), h=0.65) as q:
        a = q.ax[0]
        # posamezne Lorentzove crte vozlisc, utez = prozornost
        am = ld["aw"].max()
        for j in np.flatnonzero(ld["aw"] > 0)[::4]:
            a.plot(wf, ld["sn"][:, j], color="0.6", lw=0.3,
                   alpha=0.6 * ld["aw"][j] / am)
        a.plot(wf, ld["sp"], color=c1, ls="--", lw=0.8,
               label=r"$S(\Omega,\omega_{q_n})$")
        a.plot(wf, ld["tf"], color="k", lw=0.9,
               label=r"$T(q_n,\Omega)$")
        m = (wg >= np.abs(wf).min()) & (wg <= np.abs(wf).max())
        sg = np.sign(wf.mean()) if np.all(np.sign(wf) == np.sign(wf[0])) else 1.0
        a.scatter(sg * wg[m], sg * tg[m], color=c0, s=1.5, zorder=4,
                  label=r"$w_l$")
        a.scatter(wp, tp, color=c2, s=6, marker="s", zorder=5,
                  label="pari")
        a.plot(wp, np.full_like(wp, a.get_ylim()[0]), "|", color=c2,
               ms=4, mew=0.4)
        format_ax(a, x=wf, xlabel=r"$\Omega$",
                  ylabel=r"$A(\Omega)-A(-\Omega)$", grid=False)

        # povecava okoli para jz
        a = q.ax[1]
        w0 = wp[jz]
        sg = np.sign(w0)
        il = ld["il"][jz]
        ll = np.arange(max(il - 3, 0), il + 5)
        wz = sg * np.linspace(wg[ll[0]], wg[ll[-1]], 400)
        for wl in sg * wg[ll]:
            a.axvline(wl, color="0.85", lw=0.5, zorder=0)
        tz = np.interp(np.abs(wz), wg, tg) * sg
        a.plot(wz, ld["tf"][0] * 0 + np.interp(wz, wf, ld["tf"]),
               color="k", lw=0.9)
        a.plot(wz, tz, color=c0, lw=0.6, ls=":")
        a.scatter(sg * wg[ll], sg * tg[ll], color=c0, s=10, zorder=4)
        a.scatter(sg * wg[[il, il + 1]], sg * tg[[il, il + 1]],
                  facecolor="none", edgecolor=c0, s=40, lw=0.8, zorder=4)
        a.scatter([w0], [tp[jz]], color=c2, marker="s", s=14, zorder=5,
                  label="lookup")
        a.scatter([w0], [ld["ex"][jz]], color="k", marker="x", s=14,
                  lw=0.8, zorder=6, label="en. (20)")
        format_ax(a, x=np.sort(wz), xlabel=r"$\Omega$", grid=False)
        q.layout(add_fig_legend(q.fig, q.ax, ncol=4, top=0.84, fontsize=6))
    return q.fig, q.ax


def main():
    ap = argparse.ArgumentParser(description="E(k) rez in A(W)-A(-W)")
    ap.add_argument("--nk", type=int, default=10)
    ap.add_argument("--t", type=float, default=0.0, help="T ravnovesja")
    ap.add_argument("--mu", type=int, default=0, help="pas p0")
    ap.add_argument("--nu", type=int, default=1, help="pas k")
    ap.add_argument("--ix", type=int, default=None, help="kx indeks p0")
    ap.add_argument("--iy", type=int, default=None, help="ky indeks p0")
    ap.add_argument("--out", default=".", help="mapa za pdf")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--lut", type=int, nargs=2, default=None,
                    metavar=("QX", "QY"), help="q_n v enotah dk")
    a = ap.parse_args()

    print(a.nk)
    apply_plt_style()
    bd = eu.tb_2d(a.nk, **BAND)
    p = eu.MFPars(**MF)
    se = eu.solve_eq(bd, p, t=a.t, d=0.5, m=0.0, prog=False)
    b = pb.PhononBath(t=a.t, k=bd.k, **BAND, **MF, **BATH)
    print(f"Delta = {se.d:.6f}, m = {se.m:.6f}, ok = {se.ok}")

    ix = a.nk // 2 if a.ix is None else a.ix   # Gamma pri nk // 2
    iy = a.nk // 2 if a.iy is None else a.iy
    ip = np.ravel_multi_index((ix, iy), bd.shape)
    cd = cut_data(se.st, bd, b, a.mu, a.nu, ip)
    print(f"band fit error = {cd['err']:.2e}")

    pt = Plotter(out=a.out, show=a.show, close=not a.show, overwrite=True)
    plot_cut(pt, cd, a.mu, a.nu, name=f"cut_line_{a.mu}{a.nu}")

    if a.lut is not None:
        ld = lut_data(se.st, bd, b, a.mu, a.nu, a.lut)
        e = np.abs(ld["tp"] - ld["ex"]).max() / np.abs(ld["ex"]).max()
        print(f"lookup vs eq. (20): max rel. error = {e:.2e}")
        plot_lut(pt, ld, name=f"lut_{a.mu}{a.nu}_{a.lut[0]}{a.lut[1]}")


if __name__ == "__main__":
    main()