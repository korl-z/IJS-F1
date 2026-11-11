"""yaml -> Delta_0 and T_c -> NESS at (T1, T2). Run: python -m EI_ness.example [config.yaml]"""

import importlib
import sys
from pathlib import Path

from . import baths as B
from . import model as M

md, g, spec, cfg = B.load(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("config.yaml"))
S = importlib.import_module(f"{__package__}.{cfg['backend']}")

e0 = M.solve_eq(md, g, 0.0, **cfg["eq"])  # Delta_0
tc = M.critical_temperature(md, g)
eq = M.solve_eq(md, g, min(cfg["t"]) * tc, d=e0.d, m=e0.m)  # start: equilibrium at the colder bath
pk = B.pack(B.make(spec, cfg["t"], tc), md, g)
s = getattr(S, cfg["method"])(md, g, pk, d=eq.d, m=eq.m, **cfg[cfg["method"]])

print(f"Delta_0 = {e0.d:.6f}, T_c = {tc:.6f}, kinetic states = {2 * g.wi.size}")
print(f"T1, T2 = {cfg['t']} T_c: Delta / Delta_0 = {s.d / e0.d:.8f}, m = {s.m:.8f}, err = {s.err:.1e}, ok = {s.ok}")
# s.n: occupations (2, ni) of (alpha, beta) on the IBZ, s.mu: GGE potentials, M.unfold(s.n, g): full zone
