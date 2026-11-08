import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from matplotlib.colors import LogNorm
from tqdm import tqdm 

plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif", "font.sans-serif": ["Calibri"], "font.size": 8, "mathtext.fontset": "cm",})

def bose(x, t):
    return 1.0 / np.expm1(np.maximum(x, 1e-300) / t) if t > 0 else np.zeros_like(x)


def mf(s, gap, ta, tb, v, d, m, n=1.0):
    """Hartree shifted bands, xi, E and u, v at s (frame of ej._mf)."""
    na, nb = 0.5 * (n + m), 0.5 * (n - m)
    eah = 0.5 * gap - 2 * ta * s + v * nb - 0.5 * v
    ebh = -0.5 * gap - 2 * tb * s + v * na - 0.5 * v
    xi, av = 0.5 * (eah - ebh), 0.5 * (eah + ebh)
    ek = np.hypot(xi, d)
    u, vv = np.sqrt(0.5 * (1 + xi / ek)), np.sqrt(0.5 * (1 - xi / ek))
    return xi, np.stack((av + ek, av - ek)), u, vv


def equilibrium(s, w, gap, ta, tb, v, t, d=1.0, m=-0.99):
    """Mean field gap and imbalance at temperature t (mixing, filling 1)."""
    pbar = tqdm(range(2000), desc=f"T={t}")
    for _ in pbar:
        _, e, u, vv = mf(s, gap, ta, tb, v, d, m)
        lo, hi = e.min() - 1, e.max() + 1
        for _ in range(100):  # chemical potential for filling 1
            mu = 0.5 * (lo + hi)
            lo, hi = (mu, hi) if np.sum(w * expit(-(e - mu) / t).sum(0)) < 1 else (lo, mu)
        nq = expit(-(e - mu) / t)
        dt = v * np.sum(w * u * vv * (nq[1] - nq[0]))
        mt = np.sum(w * (u**2 - vv**2) * (nq[0] - nq[1]))
        if abs(dt - d) + abs(mt - m) < 1e-4:
            break
        d, m = d + 0.5 * (dt - d), m + 0.5 * (mt - m)
        pbar.set_postfix(d=f"{d:.5f}", m=f"{m:.5f}")
    return d, m


def occ(e, w, t):
    """Fermi occupations of both bands at filling 1."""
    lo, hi = e.min() - 1, e.max() + 1
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        lo, hi = (mu, hi) if np.sum(w * expit(-(e - mu) / t).sum(0)) < 1 else (lo, mu)
    return expit(-(e - mu) / t)
 
 
def tc(s, w, gap, ta, tb, v, lo=1e-3, hi=3.0, eps=1e-7):
    """Mean field Tc from the linearized gap equation, chi(Tc) = 1, by log bisection."""
    m = 0.0
    def chi(t, m):
        for _ in range(500):  # normal state imbalance at fixed tiny d
            _, e, u, vv = mf(s, gap, ta, tb, v, eps, m)
            nq = occ(e, w, t)
            mt = np.sum(w * (u**2 - vv**2) * (nq[0] - nq[1]))
            if abs(mt - m) < 1e-6:
                break
            m = 0.5 * (m + mt)
        return v * np.sum(w * u * vv * (nq[1] - nq[0])) / eps, m
    for _ in tqdm(range(30)):
        tm = np.sqrt(lo * hi)
        x, m = chi(tm, m)
        lo, hi = (tm, hi) if x > 1 else (lo, tm)
    return np.sqrt(lo * hi)
 

def maps(nk, it, ia, ja, sf, zs, sw, path):
    """Unfold IBZ values to the full BZ by C4v and plot Gamma_ph, Gamma_gamma and their ratio."""
    h = nk // 2
    fa, fb = np.minimum(ia, nk - ia), np.minimum(ja, nk - ja)  # mirrors fold to 0..h
    key = np.maximum(fa, fb) * (h + 1) + np.minimum(fa, fb)  # diagonal swap
    lut = np.full((h + 1) ** 2, -1)
    lut[key[it]] = np.arange(it.size)
    idx = lut[key]  # full grid point -> position in it
    q = np.linspace(-1, 1, nk, endpoint=False)
    sg = sf.reshape(nk, nk)
    lbs = (r"$\Gamma_{\mathrm{ph}}$", r"$\Gamma_{\gamma}$", r"$\Gamma_\gamma/\Gamma_{\mathrm{ph}}$")
    fig, axs = plt.subplots(1, 3, figsize=(3.47412, 0.45 * 3.47412), sharey=True)
    for ax, z, lb in zip(axs, zs, lbs):
        zm = z[idx].reshape(nk, nk)
        pz = zm[zm > 0]
        im = ax.pcolormesh(q, q, np.where(zm > 0, zm, np.nan).T, shading="nearest",
                           norm=LogNorm(vmin=np.percentile(pz, 1), vmax=pz.max()), cmap="magma")  
        im.set_edgecolor("face")
        ax.contour(q, q, sg.T, levels=sorted(set(sw) - {2.0}) or [sw[0]], colors="w",
                   linewidths=0.4, linestyles="--")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$k_x/\pi$", fontsize=6)
        ax.tick_params(which="both", direction="in", labelsize=5)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", location="top", pad=0.02, fraction=0.06)
        cb.set_label(lb, fontsize=6, labelpad=2)
        cb.ax.tick_params(labelsize=5)
        cb.ax.xaxis.set_minor_formatter(plt.NullFormatter())  
    axs[0].set_ylabel(r"$k_y/\pi$", fontsize=6)
    # fig.savefig(path, bbox_inches="tight")
    # plt.show()


# smin=1.25, smax=1.9 za bcs primer
# smin=1.7, smax=2.0 za BEC

def run(nk=200, smin=1.7, smax=2.0, t1=0.8, t2=0.2, gap=4.1, ta=1.0, tb=-1.0, v=4.0,
        amp_ph=1 / (2 * 598.0 * 200), tpa=-1.0e-3, tpb=1.0e-3, cs=0.05, gam_ph=0.05, qd=np.sqrt(4 * np.pi),
        w0=0.145, gam_ga=0.05, c=3945.0, amp_ga=4 / 137.036 * 0.145, path="rates_xi_BEC2.pdf"):
    # IBZ 0 <= ky <= kx <= pi
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False) #full grid
    kx, ky = (a.ravel() for a in np.meshgrid(k, k, indexing="ij")) 
    sf = np.cos(kx) + np.cos(ky) 
    ia, ja = (np.mod(a - nk // 2, nk).ravel() for a in np.meshgrid(np.arange(nk), np.arange(nk), indexing="ij")) 
    tri = (ia <= nk // 2) & (ja <= ia)  # 0 <= ky <= kx <= pi, pi = -pi (mod G)
    wf = np.full(kx.size, 1.0 / kx.size)

    t_c = tc(sf, wf, gap, ta, tb, v)
    print(t_c)
    print(t_c * 0.1 /(8.617 * 1e-5))

    d, m = equilibrium(sf, wf, gap, ta, tb, v, t=t2 * t_c)

    xf, ef, uf, vf = mf(sf, gap, ta, tb, v, d, m)
    af, bf = np.concatenate((uf, vf)), np.concatenate((-vf, uf))  # alpha, beta coefficients
    eall = ef.ravel()
    kxa, kya = np.tile(kx, 2), np.tile(ky, 2)

    it = np.flatnonzero(tri)  # alpha targets: all IBZ states (maps), window below (xi plot)
    win = (sf[it] >= smin) & (sf[it] <= smax)
    gph, gpi, gph2, gpi2 = np.zeros(it.size), np.zeros(it.size), np.zeros(it.size), np.zeros(it.size)
    for c0 in range(0, it.size, 64):  # out rate of alpha_k to all 2N states
        i = it[c0:c0 + 64, None]
        qx = np.mod(kxa[None] - kx[i] + np.pi, 2 * np.pi) - np.pi
        qy = np.mod(kya[None] - ky[i] + np.pi, 2 * np.pi) - np.pi
        q = np.hypot(qx, qy)
        om = np.maximum(cs * q, 1e-30)
        de = eall[None] - ef[0, i]  # E_dest - E_alpha_k
        wa = np.abs(de)
        c1 = 2 * (tpa * uf[i] * af[None] + tpb * (-vf[i]) * bf[None])

        vtx1 = (np.sin(kxa[None]) - np.sin(kx[i])) ** 2 + (np.sin(kya[None]) - np.sin(ky[i])) ** 2
        # vtx2 = 4 - 2 * (np.cos(qx) + np.cos(qy)) - 0.5 * ((np.cos(kxa[None]) + np.cos(kya[None])) - (np.cos(kx[i]) + np.cos(ky[i]))) ** 2
        vtx2 = 4 - ((np.cos(kxa[None]) + np.cos(kya[None])) * (np.cos(kx[i]) + np.cos(ky[i])) -  np.sqrt((4 -(np.cos(kxa[None]) + np.cos(kya[None]))**2) *(4 - (np.cos(kx[i]) + np.cos(ky[i]))**2) )) - 0.5 * ((np.cos(kxa[None]) + np.cos(kya[None])) - (np.cos(kx[i]) + np.cos(ky[i]))) ** 2 
        vtx2 = 0.5 * vtx2

        d1, d2 = (wa - om) ** 2 + gam_ph**2, (wa + om) ** 2 + gam_ph**2
        sp = 8 * gam_ph * wa * om / (d1 * d2)
        nb = bose(wa, t2 * t_c)
        bfac = np.where(de > 0, sp * nb, sp * (1 + nb))  # absorption up, emission down
        bfac = np.where(wa < 1e-12, 8 * gam_ph * om * t2 * t_c / (om**2 + gam_ph**2) ** 2, bfac)

        r = amp_ph * c1**2 * vtx1 / om * bfac * (q <= qd) / kx.size
        r2 = amp_ph * c1**2 * vtx2 / om * bfac * (q <= qd) / kx.size

        r[np.arange(i.size), i[:, 0]] = 0.0  # no self term
        r2[np.arange(i.size), i[:, 0]] = 0.0  # no self term
        gph[c0:c0 + 64] = r.sum(1)
        gph2[c0:c0 + 64] = r2.sum(1)

        gpi[c0:c0 + 64] = r[:, kx.size:].sum(1)  # alpha -> beta part
        gpi2[c0:c0 + 64] = r2[:, kx.size:].sum(1)  # alpha -> beta part


    # photons, analytic k diagonal rate out of alpha_k (emission alpha -> beta)
    om = ef[0, it] - ef[1, it]
    z4 = (om**2 + gam_ga**2) ** 2
    kk = ((1 + w0**2 * (om**2 - gam_ga**2) / z4) * (np.arctan((om - w0) / gam_ga) + np.arctan((om + w0) / gam_ga))
          + gam_ga * om * w0**2 / z4 * np.log(((om - w0) ** 2 + gam_ga**2) * ((om + w0) ** 2 + gam_ga**2) / w0**4))
    jj1 = amp_ga / (2 * np.pi * c**2) * (np.sin(kx[it]) ** 2 + np.sin(ky[it]) ** 2) * (2 * (ta - tb) * uf[it] * vf[it]) ** 2 * kk
    jj2 = amp_ga / (2 * np.pi * c**2) *  (2 - 0.5 * (np.cos(kx[it]) + np.cos(ky[it])) ** 2) * (2 * (ta - tb) * uf[it] * vf[it]) ** 2 * kk
    gga1 = jj1 * (1 + bose(om, t1 * t_c))
    gga2 = jj2 * (1 + bose(om, t1 * t_c))

    rat = np.where(gph > 0, gga1 / np.maximum(gph, 1e-300), 0.0)
    maps(nk, it, ia, ja, sf, (gph, gga1, rat), (smin, smax), path.replace(".pdf", "_bz.pdf"))
    x, gph, gpi, gga1, gga2, gph2, gpi2 = xf[it][win], gph[win], gpi[win], gga1[win], gga2[win], gph2[win], gpi2[win]
    cl = plt.cm.Set1.colors
    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, figsize=(3.47412, 1.1 * 3.47412))
    ax.scatter(x, gph, alpha = 0.6, s=1, color=cl[0], label=r"$\Gamma_{\mathrm{ph}}$")
    ax.scatter(x, gpi, alpha = 0.6, s=1, color=cl[2], label=r"$\Gamma_{\mathrm{ph}}^{\alpha\to\beta}$")
    ax.scatter(x, gph2,alpha = 1,  s=1, color=cl[4], label=r"$\Gamma_{\mathrm{ph}}$ (approx)")
    ax.scatter(x, gpi2,alpha = 1,  s=1, color=cl[6], label=r"$\Gamma_{\mathrm{ph}}^{\alpha\to\beta}$ (approx)")
    ax.scatter(x, gga1,alpha = 0.6,  s=1, color=cl[1], label=r"$\Gamma_{\gamma}$")
    ax.scatter(x, gga2,alpha = 1 ,  s=1, color=cl[3], label=r"$\Gamma_{\gamma}$ (approx.)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Gamma\ [t_a/\hbar]$")
    bx.scatter(x, gga1 / gph,  alpha = 0.5 , s=1, color=cl[3], label=r"$\Gamma_\gamma/\Gamma_{\mathrm{ph}}$")
    bx.scatter(x, gga2 / gph2, alpha = 1,  s=1, color=cl[6], label=r"$\Gamma_\gamma/\Gamma_{\mathrm{ph}}$ (approx.)")
    # # xi model (solid) and xi = 0 expansion (dashed), anchored at xi = 0
    # xm, mph, mpi, mga = xi_model(d, m, smin, smax, t1 * t_c, t2 * t_c, gap, ta, tb, v, amp_ph, tpa, tpb,
    #                              cs, qd, gam_ph, w0, gam_ga, c, amp_ga)
    # o = np.argsort(xm)
    # xm, mph, mpi, mga = xm[o], mph[o], mpi[o], mga[o]
    # lph, lga = local(xm, d, t2 * t_c, (np.interp(0, xm, mph), np.interp(0, xm, mga)))
    # for y, cc in ((mph, cl[0]), (mpi, cl[2]), (mga, cl[1])):
    #     ax.plot(xm, y, color=cc, lw=0.8)
    # ax.plot(xm, lph, "--", color=cl[0], lw=0.8)
    # ax.plot(xm, lga, "--", color=cl[1], lw=0.8)
    # bx.plot(xm, mga / mph, color=cl[3], lw=0.8, label=r"$\xi$ model")
    # bx.plot(xm, lga / lph, "--", color=cl[3], lw=0.8, label=r"$\xi\to0$")
    # bx.set_yscale("log")
    bx.set_xlabel(r"$\xi_{\mathbf{k}}$")
    bx.set_ylabel(r"$\Gamma_\gamma/\Gamma_{\mathrm{ph}}$")
    pos = lambda *z: np.concatenate([q[q > 0] for q in z])
    for a, z in ((ax, pos(gph, gpi, gga1, gga2, gph2, gpi2)), (bx, pos(gga1 / gph, gga2 / gph2))):
        a.set_ylim(max(1.0e-16, 0.5 * z.min()), 2.0 * z.max())  # log autoscale of scatter is unreliable
        a.set_xlim(x.min(), x.max())
        a.grid(alpha=0.3)
        a.legend(markerscale=4, fontsize=6)
        a.tick_params(which="both", direction="in")
    plt.tight_layout()
    # fig.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"d = {d:.5f}, m = {m:.5f}, N_targets = {it.size}, "
          f"median ratio = {np.median(gga1 / gph):.2e}")
    return x, gph, gpi, gga1, gga2, gph2, gpi2


if __name__ == "__main__":
    run()