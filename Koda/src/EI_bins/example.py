"""Minimal usage example: one steady state and one T1 sweep.

Run from Koda/src:  python -m EI_bins.example
"""

import time
import warnings

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import EI_bins as eb
from EI_bins.checks import check_db

warnings.simplefilter("once")

# model, as in config_phonons.yaml (gap, ta, tb, v)
bd = eb.Band(gap=2.0, ta=1.0, tb=-1.0)
mf = eb.MF(v=4.4, n=1.0)

# equilibrium scales on a fine shell grid (Tc by bisection on d)
ref = eb.make_shells(257)
d0 = eb.solve_eq(ref, bd, mf, 1e-3, 1.0, -0.6).d
lo, hi = 0.05, 1.5
for _ in range(40):
    t = 0.5 * (lo + hi)
    e = eb.solve_eq(ref, bd, mf, t, 0.5 * d0, -0.6)
    lo, hi = (t, hi) if (e.ok and e.d > 1e-4) else (lo, t)
tc = 0.5 * (lo + hi)
print(f"d0 = {d0:.6f}, Tc = {tc:.6f}")

# Einstein phonons, same geometry for both baths
w0 = 2.1
c = ((1.0, 2.0), (2.0, 1.0))
sh = eb.make_shells(eb.auto_ns(bd, min(d0, tc, w0)))
A = eb.pair_kernel(sh, lam=0.01, qd=4.44288, cache="kernels")
print("ns =", sh.ns)


def baths(t1, t2=0.5 * tc):
    return [eb.Phonon(t1, 1.0, w0, 0.01, 4.44288, c, name="bath_1"),
            eb.Phonon(t2, 1.0, w0, 0.01, 4.44288, c, name="bath_2")]




print("T1 = T2 check:", check_db(sh, bd, mf, baths(0.5 * tc)[0], A, 0.5 * tc))

ts = time.perf_counter()
# # s = eb.solve_ness(sh, bd, mf, baths(1.0 * tc), A=A)
# print(f"T1 = Tc: d = {s.d:.6f}, m = {s.m:.6f}, {s.branch}, stable = {s.stable}, "
#       f"rate = {s.rate:.2e}, {time.perf_counter() - ts:.2f} s")

xs = np.linspace(0.01, 1.6, 24) * tc
ts = time.perf_counter()
# row_cpu = eb.sweep(sh, bd, mf, baths, xs, A=A)
dt_cpu = time.perf_counter() - ts

ts = time.perf_counter()
row_jax = eb.sweep_jax(sh, bd, mf, baths, xs, A=A)
dt_jax = time.perf_counter() - ts
print(f"CPU sweep: {dt_cpu:.2f} s for {len(xs)} points")
print(f"JAX sweep: {dt_jax:.2f} s for {len(xs)} points (includes JIT compilation)")

# da_cpu = np.array([r.d for r in row_cpu])
da_jax = np.array([r.d for r in row_jax])
# print(f"Maximum gap difference: {np.max(np.abs(da_cpu - da_jax)):.3e}")
# for x, rc, rj in tqdm(zip(xs, row_cpu, row_jax), total=len(xs), desc="Sweep results"):
#     print(f"  T1/Tc = {x / tc:.2f}  CPU: d = {rc.d:.5f}, m = {rc.m:.5f}, {rc.branch}"
#           f"  JAX: d = {rj.d:.5f}, m = {rj.m:.5f}, {rj.branch}")

set1_list = plt.get_cmap("Set1").colors
h = 0.6
fig, ax = plt.subplots(figsize=(3.47412, h * 3.47412))
# ax.plot(xs / tc, da_cpu, color=set1_list[0], label="CPU")
ax.plot(xs / tc, da_jax, color=set1_list[1], linestyle="--", label="JAX")
ax.set_xlim(xs[0] / tc, xs[-1] / tc)
ax.set_xlabel(r"$T_1 / T_c$")
ax.set_ylabel(r"$\Delta$")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()

def tc_line(thot, tc):
    """Cold-bath temperature on T*(eps) = Tc, as a function of the hot-bath temperature."""
    nc = 1.0 / np.expm1(w0 / tc)                     # N(eps, Tc)
    nh = 1.0 / np.expm1(w0 / thot)                   # N(eps, T_hot)
    nl = (nc -  nh) / (2)                  # cold occupation on the line
    return np.where(nl > 0.0, w0 / np.log1p(1.0 / nl), np.nan)

t1a = np.linspace(0.01, 1.8, 51) * tc
t2a = np.linspace(0.01, 1.8, 51) * tc
da = np.empty((t2a.size, t1a.size))

for j, t2 in tqdm(enumerate(t2a), total=t2a.size, desc="T2 rows"):
    row = eb.sweep_jax(
        sh, bd, mf, lambda t1: baths(t1, t2), t1a,
        A=A, stab=False,
    )
    da[j] = [s.d for s in row]

h = 0.8
fig, ax = plt.subplots(figsize=(3.47412, h * 3.47412))
im = ax.pcolormesh(t1a / tc, t2a / tc, da, shading="nearest")
im.set_edgecolor("face")

xh = np.linspace(t1a[0], t1a[-1], 400)                  # T1/Tc (hot)
y0 = tc_line(xh, tc)     

ax.plot(xh / tc, y0 / tc, color="white")
ax.set_xlim(t1a[0] / tc, t1a[-1] / tc)
ax.set_ylim(t2a[0] / tc, t2a[-1] / tc)
ax.set_xlabel(r"$T_1/T_c$")
ax.set_ylabel(r"$T_2/T_c$")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$\Delta$", rotation=90, labelpad=5)
plt.tight_layout()
plt.show()