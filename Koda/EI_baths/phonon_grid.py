import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def mom_grid(bd, a0=1.0):
    """Return q[i,j] = wrap(k[i] - k[j]), destination first."""
    k = np.asarray(bd.k, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2 or not len(k) or not np.isfinite(k).all():
        raise ValueError("bd.k must have shape (N, 2) with finite entries")
    if not np.isfinite(a0) or a0 <= 0:
        raise ValueError("a0 must be finite and positive")
    per = 2 * np.pi / a0
    dk = k[:, None, :] - k[None, :, :]
    q = (dk + per / 2) % per - per / 2
    tol = 128 * np.finfo(float).eps * max(per, np.max(np.abs(k)))
    ii = np.arange(len(k))
    np.testing.assert_allclose(q[ii, ii], 0, atol=tol, rtol=0)
    z = (q + q.swapaxes(0, 1) + per / 2) % per - per / 2
    np.testing.assert_allclose(z, 0, atol=tol, rtol=0)
    z = (dk - q) / per
    np.testing.assert_allclose(z, np.rint(z), atol=tol / per, rtol=0)
    if np.any(q < -per / 2 - tol) or np.any(q >= per / 2 + tol):
        raise AssertionError("momentum transfer outside the first BZ")
    return q


def ene_grid(bd, st, h, ed=None):
    """Assign both bands to shared half-open bins; keep empty bins."""
    e = np.asarray(st.e, dtype=float)
    w = np.asarray(bd.w, dtype=float)
    nk = len(bd.k)
    if e.shape != (2, nk) or w.shape != (nk,):
        raise ValueError("st.e and bd.w must have shapes (2, N) and (N,)")
    if not np.isfinite(e).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("energies and weights must be finite; weights nonnegative")
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12, rtol=0)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if ed is None:
        lo = int(np.floor(e.min() / h)) - 1
        hi = int(np.floor(e.max() / h)) + 2
        ed = h * np.arange(lo, hi + 1, dtype=float)
    else:
        ed = np.array(ed, dtype=float, copy=True)
    if ed.ndim != 1 or ed.size < 2 or not np.isfinite(ed).all():
        raise ValueError("ed must be a finite array of bin edges")
    tol = 128 * np.finfo(float).eps * max(h, np.max(np.abs(ed)))
    if np.any(np.diff(ed) <= 0):
        raise ValueError("bin edges must increase strictly")
    np.testing.assert_allclose(np.diff(ed), h, atol=tol, rtol=1e-12)
    if e.min() < ed[0] or e.max() >= ed[-1]:
        raise ValueError("spectrum outside fixed bins; supply a wider ed array")
    ec = (ed[:-1] + ed[1:]) / 2
    ix = np.searchsorted(ed, e, side="right") - 1
    eb = ec[ix]
    wb = np.empty((2, ec.size))
    ct = np.empty((2, ec.size), dtype=int)
    for a in tqdm(range(2), desc="Energy bins", leave=False):
        wb[a] = np.bincount(ix[a], weights=w, minlength=ec.size)
        ct[a] = np.bincount(ix[a], minlength=ec.size)
    np.testing.assert_allclose(wb.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(ct.sum(axis=1), nk)
    if np.max(np.abs(eb - e)) > h / 2 + tol:
        raise AssertionError("energy rounding exceeds half a bin")
    return dict(e=e, h=float(h), ed=ed, ec=ec, ix=ix, eb=eb, wb=wb, ct=ct)


def plot_grid(q, z, set1_list, a0=1.0):
    """Plot both momentum components, energy assignments, and bin weights."""
    nk = q.shape[0]
    if nk < 2:
        raise ValueError("at least two momenta are needed for the table plots")
    ii = np.arange(nk)
    h = 0.8
    fig, ax = plt.subplots(1, 2, figsize=(2*3.47412, h * 3.47412))
    lb = (r"$q_x a_0/\pi$", r"$q_y a_0/\pi$")
    for a in tqdm(range(2), desc="Momentum plots", leave=False):
        im = ax[a].pcolormesh(ii, ii, q[:, :, a] * a0 / np.pi,
                             shading="nearest", cmap="RdBu_r", vmin=-1, vmax=1,
                             rasterized=True)
        im.set_edgecolor("face")
        cbar = plt.colorbar(im, ax=ax[a])
        cbar.set_label(lb[a], rotation=90, labelpad=5)
        ax[a].set(xlim=(ii[0], ii[-1]), ylim=(ii[0], ii[-1]),
                  xlabel=r"$j\;\mathrm{(source)}$", ylabel=r"$i\;\mathrm{(destination)}$")
    plt.tight_layout()
    f1 = fig

    fig, ax = plt.subplots(1, 2, figsize=(2*3.47412, h* 3.47412))
    lb = (r"$\alpha$", r"$\beta$")
    for a in tqdm(range(2), desc="Energy plots", leave=False):
        ax[0].scatter(z["e"][a], z["eb"][a], s=9, color=set1_list[a],
                      label=lb[a], alpha=0.7)
        ax[1].stairs(z["wb"][a] / z["h"], z["ed"], color=set1_list[a],
                     linewidth=1.1, label=lb[a])
    lo, hi = z["e"].min(), z["e"].max()
    if lo == hi:
        lo, hi = z["ed"][0], z["ed"][-1]
    ax[0].plot([lo, hi], [lo, hi], ":", color="black", linewidth=0.8,
               label=r"$\bar E=E$")
    ax[0].set(xlim=(lo, hi), xlabel=r"$E_{\lambda k}$", ylabel=r"$\bar E_{\lambda k}$")
    ax[1].set(xlim=(z["ed"][0], z["ed"][-1]),
              xlabel=r"$E$", ylabel=r"$W_{\lambda a}/h$")
    for aa in tqdm(ax, desc="Axes", leave=False):
        aa.grid(alpha=0.3)
        aa.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return f1, fig
