"""Short self check (seconds). Run: python -m EI_ness.check"""

import warnings

import numpy as np
from scipy.special import expit

from . import baths as B
from . import cpu as C
from . import model as M

warnings.simplefilter("ignore")
md, g = M.Model(2.0, 1.0, -1.0, 4.4), M.grid(16)
e0 = M.solve_eq(md, g, 0.0, d=1.0)
tc = M.critical_temperature(md, g)
eq = M.solve_eq(md, g, 0.3 * tc, d=e0.d, m=e0.m)
sets = {
    "toy": lambda t: [B.Toy(t[0], kap=(0.1, 0.2, 0.1)), B.Toy(t[1], kap=(0.1, 0.2, 0.1))],
    "cavity + phonon": lambda t: [B.Cavity(t[0], 0.1, 0.01), B.Phonon(t[1], 0.13, 0.1, 4.2e-5, 1.0, -1.0)],
    "photon + phonon": lambda t: [B.Photon(t[0], 10.0, 3.0, 0.15, amp=1.0, dip=2.0),
                                  B.Phonon(t[1], 1.6, 0.15, 1.0, 1.0, -1.0, w0=0.01)],
}
try:
    from . import gpu as G
except ImportError:
    G = None
    print("jax not found: gpu checks skipped")

for name, mk in sets.items():
    # equal temperatures: detailed balance and the Fermi state
    t = 0.3 * tc
    pk = B.pack(mk((t, t)), md, g)
    db = C.check_db(C.rates(md, g, eq.st, pk), eq.st.e.reshape(-1), t)
    s = C.solve(md, g, pk, d=eq.d, m=eq.m)
    dn = np.abs(s.n - expit(-(eq.st.e - eq.mu) / t)).max()
    # two temperatures: both solvers, both backends
    pk = B.pack(mk((1.5 * tc, t)), md, g)
    a = C.solve(md, g, pk, d=eq.d, m=eq.m)
    f = C.solve_fixed(md, g, pk, d=eq.d, m=eq.m, tol=1.0e-7)
    out = (f"{name:16s} detailed balance {db:.0e} | T1 = T2: max |n - f| {dn:.0e} | T1 = 1.5 Tc: d / d0 "
           f"solve {a.d / e0.d:.7f} ({len(a.hist['d'])} builds, ok {a.ok}) fixed {f.d / e0.d:.7f} (ok {f.ok})")
    if G is not None:
        rc = C.rates(md, g, eq.st, pk)
        rg = np.asarray(G.rates(md, G.dev(g), G.mf(md, G.dev(g), eq.d, eq.m), G.dev(pk)))
        ga = G.solve(md, g, pk, d=eq.d, m=eq.m)
        gf = G.solve_fixed(md, g, pk, d=eq.d, m=eq.m, tol=1.0e-7)
        out += (f" | gpu: rates {np.abs(rc - rg).max() / rc.max():.0e}, solve {abs(ga.d - a.d):.0e},"
                f" fixed {abs(gf.d - f.d):.0e}")
    print(out)
