import numpy as np

def gamma_idx(bd):
    """Index of the grid point closest to Gamma."""
    k = np.asarray(bd.k)
    return int(np.argmin(np.sum(k * k, axis=1)))


def tri_idx(bd):
    """Indices inside the Gamma-X-M triangle."""
    k = np.asarray(bd.k)
    x = k[:, 0]
    y = k[:, 1]

    q = np.unique(x)
    dq = np.min(np.diff(q))
    z = q[np.argmin(np.abs(q))]
    ep = 0.25 * dq

    ok = (
        (x <= z + ep)
        & (y <= z + ep)
        & (y >= x - ep)
    )
    return np.flatnonzero(ok)


def gap_info(bd, st):
    """Relevant signed, direct, and indirect band gaps."""
    ea = np.asarray(bd.ea)
    eb = np.asarray(bd.eb)
    eah = np.asarray(st.eah)
    ebh = np.asarray(st.ebh)
    ed = np.asarray(st.e)

    ig = gamma_idx(bd)
    it = tri_idx(bd)

    dd = ed[0] - ed[1]
    im = it[np.argmin(dd[it])]

    return {
        "bare_g": float(ea[ig] - eb[ig]),
        "hartree_g": float(eah[ig] - ebh[ig]),
        "diag_g": float(dd[ig]),
        "diag_min": float(dd[im]),
        "diag_ind": float(
            np.min(ed[0, it]) - np.max(ed[1, it])
        ),
        "bare_ind": float(
            np.min(ea[it]) - np.max(eb[it])
        ),
        "hartree_ind": float(
            np.min(eah[it]) - np.max(ebh[it])
        ),
        "k_min": np.asarray(bd.k[im]),
    }