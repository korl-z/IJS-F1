import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ei_open_loop import diag, fd, gam_db, solve


def scan(t1a, t2a, x, ea, eb, g, uh, mkgs, n0=1.0, d0=0.2, m0=0.0,
         dt=0.05, td=1.0, tm=1.0, tol=1.0e-7, nmax=50000,
         sym=False):
    """Scan the steady EI gap over two bath temperatures."""
    t1a = np.asarray(t1a, dtype=float)
    t2a = np.asarray(t2a, dtype=float)
    da = np.zeros((t2a.size, t1a.size))
    ma = np.zeros_like(da)
    er = np.zeros_like(da)
    cv = np.zeros_like(da, dtype=bool)

    e0, _, _ = diag(ea, eb, d0, m0, uh=uh, n0=n0)
    ni = fd(e0, max(t1a[0], t2a[0]), n0)
    n = ni.copy()
    d = d0
    m = m0

    def one(i, j, n, d, m):
        s = solve(
            x,
            ea,
            eb,
            g=g,
            n=n,
            gs=mkgs(t1a[i], t2a[j]),
            d=max(abs(d), 1.0e-5),
            m=m,
            uh=uh,
            n0=n0,
            dt=dt,
            td=td,
            tm=tm,
            nmax=nmax,
            tol=tol,
            chk=100,
        )
        return s

    def put(i, j, s):
        da[j, i] = abs(s["d"])
        ma[j, i] = s["m"]
        er[j, i] = s["err"]
        cv[j, i] = s["ok"]

    if sym:
        if t1a.shape != t2a.shape or not np.allclose(t1a, t2a):
            raise ValueError("sym requires identical temperature grids")

        for j in tqdm(range(t2a.size)):
            for i in range(j, t1a.size):
                s = one(i, j, ni.copy(), d0, m0)
                put(i, j, s)
                put(j, i, s)
        return {"t1": t1a, "t2": t2a, "d": da, "m": ma,
                "err": er, "ok": cv}

    for j, t2 in tqdm(enumerate(t2a)):
        js = range(t1a.size) if j % 2 == 0 else range(t1a.size - 1, -1, -1)

        for i in js:
            s = one(i, j, n, d, m)
            n = s["n"]
            d = s["d"]
            m = s["m"]
            put(i, j, s)

    return {"t1": t1a, "t2": t2a, "d": da, "m": ma, "err": er, "ok": cv}


def plot(z, fn):
    """Plot the gap as a temperature-temperature color map."""
    fig, ax = plt.subplots(figsize=(3.47412, 0.82 * 3.47412))
    im = ax.pcolormesh(z["t1"], z["t2"], z["d"], shading="nearest",
                       cmap="magma")
    im.set_edgecolor("face")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta$", rotation=90, labelpad=5)
    ax.set_xlabel(r"$T_1$")
    ax.set_ylabel(r"$T_2$")
    ax.set_xlim(z["t1"][0], z["t1"][-1])
    ax.set_ylim(z["t2"][0], z["t2"][-1])
    plt.tight_layout()
    fig.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    nk = 41
    x = (np.arange(nk) + 0.5) / nk
    gg = -0.4
    rr = 0.7
    ea = 0.5 * gg + x
    eb = -0.5 * gg - rr * x

    t1a = np.linspace(0.04, 0.80, 21)
    t2a = np.linspace(0.04, 0.80, 21)

    c1 = 0.3
    c2 = 0.3
    wc = 2.0

    def mkgs(t1, t2):
        return [
            gam_db(t=t1, c=c1, wc=wc),
            gam_db(t=t2, c=c2, wc=wc),
        ]

    z = scan(
        t1a,
        t2a,
        x,
        ea,
        eb,
        g=1.5,
        uh=0.5,
        mkgs=mkgs,
        n0=1.0,
        d0=0.2,
        m0=0.0,
        dt=1.0,
        td=0.2,
        tm=0.2,
        tol=1.0e-5,
        sym=True,
    )

    np.savez("ei_phase_map.npz", **z)
    plot(z, "ei_phase_map.pdf")
    print(f"converged={z['ok'].mean():.3f} max_err={z['err'].max():.3e}")
