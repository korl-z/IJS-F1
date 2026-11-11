# EI_ness

Steady state (NESS) of the two band excitonic insulator coupled to baths: order parameter d,
imbalance m and quasiparticle occupations n for given baths and temperatures.
Independent of the old `src/EI` code. Equation numbers refer to `progress_notes`.

```
EI_ness/
  model.py     grid (C4v irreducible BZ + s window), mean field, equilibrium, T_c     numpy
  baths.py     bath definitions, cell averaged line tables, yaml loader               numpy
  cpu/rates.py     rate matrix R[dest, src]                                           numpy
  cpu/solvers.py   solve (Newton), solve_fixed (Anderson)                             numpy / scipy
  gpu/rates.py     same functions, same names                                         jax
  gpu/solvers.py
  config.yaml  example.py  check.py
```

`cpu/` never imports jax. `gpu/` mirrors `cpu/` line by line (see "JAX status").

## Use

```python
from EI_ness import model as M, baths as B
from EI_ness import cpu as S            # or: from EI_ness import gpu as S

md = M.Model(gap=2.0, ta=1.0, tb=-1.0, v=4.4)        # v = 7.8: BEC case with the same Delta_0
g = M.grid(64)                                       # or M.grid(150, win=(1.0, 2.0))
e0 = M.solve_eq(md, g, 0.0, d=1.0)                   # Delta_0
tc = M.critical_temperature(md, g)

bs = [B.Cavity(t=1.5 * tc, w0=0.1, gam=0.01),                               # analytic photons
      B.Phonon(t=0.3 * tc, cs=0.13, gam=0.1, amp=4.2e-5, tpa=1.0, tpb=-1.0)]  # phonons on the grid
# toy model instead: bs = [B.Toy(t1, kap=(0.1, 0.2, 0.1)), B.Toy(t2, kap=(0.1, 0.2, 0.1))]
pk = B.pack(bs, md, g)                               # static tables, cached

eq = M.solve_eq(md, g, 0.3 * tc, d=e0.d, m=e0.m)     # start
s = S.solve(md, g, pk, d=eq.d, m=eq.m)               # s.d, s.m, s.n (2, ni), s.mu, s.err, s.ok
s2 = S.solve(md, g, pk2, d=max(s.d, 0.2 * e0.d), m=s.m, mu=s.mu)   # next point of a scan
nfull = M.unfold(s.n, g)                             # (2, nk, nk) for plots
```

yaml: `python -m EI_ness.example config.yaml`. The bath set is chosen with `use: toy | real | grid`,
the temperatures with `t: [T1 / Tc, T2 / Tc]`, the backend with `backend: cpu | gpu`.
`B.make(spec, ts, tu)` builds the bath list from the yaml dicts (`ts` overrides the temperatures,
`tu` is their unit), so a scan only calls `B.make` and `B.pack` per point; the tables are cached.

`python -m EI_ness.check` runs a self check in a few seconds (also compares gpu with cpu if jax is there).

## model.py

| function | does |
|---|---|
| `grid(nk, win)` | Gamma centred grid k = 2 pi i / nk, reduced to 0 <= ky <= kx <= pi. Stores the integer coordinates `ki`, the 8 C4v images `kg`, `s = cos kx + cos ky`, orbit weights `w`, the kinetic states `wi` (s window) and sine tables. |
| `unfold(a, g)` | IBZ array to the full nk x nk zone. |
| `mf(md, g, d, m)` | Bogoliubov energies and u, v (eq. 33, 44, 45), frame shifted by -h / 2. |
| `targets` | d*, m*, filling from occupations (eq. 37, 42, 43). |
| `fermi`, `solve_eq`, `solve_normal`, `pair_lambda`, `critical_temperature`, `bec_zero`, `bec_lam1` | equilibrium solvers, same algorithms as `ei_unified` (eq. 46 - 48). |
| `gap_info` | bare, Hartree and quasiparticle gaps (from `ei_utils`). |

## baths.py

| bath | rate |
|---|---|
| `Toy(t, kap, wc, orb, mu, rate)` | factorizable rates of `ei_jax_2` (`rate="product"` or `"db"`), unchanged. |
| `Phonon(t, cs, gam, amp, tpa, tpb, w0, qd, nsub, box)` | Peierls vertex eq. 97 - 99 with D = 0, `amp = 1 / (2 M)` (eq. 282), omega = hypot(w0, cs q). |
| `Photon(t, c, w0, gam, amp, dip, qd, nsub, box)` | eq. 258: TE1 + TM1 with C, z dipole TM2 with D. `amp = None`: 4 alpha w0 (eq. 280). |
| `Cavity(t, w0, gam, c, amp)` | q integrated photons, vertical transitions, eq. 286, 287. |

Any mix of kinds is allowed in one list (toy and grid baths can now be combined).
`pack(bs, md, g)` turns the list into plain arrays; the band parameters come from `md`, the
grid from `g`, so a bath only holds bath physics.
`ltab` builds the cell averaged line tables of eq. 274, 275 (see changes 3 and 4).

## cpu/rates.py, gpu/rates.py

`rates(md, g, st, pk)` returns R[dest, src] on the kinetic states (alpha, beta) x `g.wi`.
The solvers use a = R (w n), b = R^T (w (1 - n)), dn = (1 - n) a - n b (eq. 184, 192).
`therm` is Omega N(Omega) (eq. 188, 189), `_toy`, `_lin`, `_cav` are the three bath kinds,
`_look` interpolates the line tables, `check_db` tests eq. 190.

## cpu/solvers.py, gpu/solvers.py

Everything is solved in the GGE potentials mu, n = 1 / (1 + e^mu) (eq. 191), so both n and 1 - n
keep full precision.

`occ(r, ww, pn, mu, dead)`: exact stationary state of frozen rates by Newton. Rows
mu_i - log(b_i / a_i) + kap_band = 0 (eq. 192 in log form), filling, and a pair row
log(interband flux into alpha / out of alpha) = 0. The two kap absorb the redundancy from the
conservation of N_alpha and N_beta by intraband rates and vanish at the solution. The pair row
is what fixes the slow pair number mode, without cancellation against the large intraband terms.

`solve(md, g, pk, d, m, mu)`: root of F(log d, m) = (d* / d - 1, m* - m), each evaluation is one
rate build and one `occ`.
1. Newton in 2 variables (finite difference Jacobian, Broyden updates), accepted only if the root
   is stable under dz / dt = F.
2. Otherwise nested bracketing (`_root`) that follows the flow from the start: m(d) inside,
   log d outside. Ends in the normal state (d = 0 returned) if d* < d all the way down.

`solve_fixed(md, g, pk, d, m, mu, mix, mem, chk, ...)`: fixed point mu -> log(b / a) with Anderson
acceleration (`mem = 0`: Picard), rates rebuilt every `chk` steps, pair number corrected after
every chunk (`_pfix`).

`resid(md, g, r, mu, d, m)`: the stopping measure of both solvers, scale free:
max |gain - loss| / (gain + loss) over states, net / gross interband flux, |d* - d| / d, |m* - m|.

## Changes with respect to the old code and the notes

1. Only the irreducible BZ. R is the average over the 8 C4v images of the source, which equals the
   old orbit average. Its diagonal is set to zero: scattering inside one orbit and band cancels in dn.
   The grid is Gamma centred, so odd nk works; for even nk the points are those of `tb_2d`.
2. One energy frame (shifted by -h / 2) everywhere, also in equilibrium: `Eq.mu` is the old mu - h / 2.
3. Lines are always Lorentzians (eq. 272). The hat / delta branch is gone, `gam <= 0` is an error.
   The energy axis of the tables is w = omega_q + gam sinh(hh i) instead of a uniform step gam / 8:
   the table size grows like log(1 / gam) (511 points per q at gam = 1e-4, where the uniform table needs 1.3e6).
   `em` is set from `em_bound` automatically.
4. Deviation from eq. 275 (default `box = 1`, `box = 0` gives the plain sum): the Lorentzian of
   every tent node is averaged analytically over the omega interval that the node stands for.
   For gam below the node spacing 2 cs dk / nsub the plain sum is a comb of nsub^2 peaks; the box
   average is smooth for any gam > 0. Against a converged reference (nsub = 64) the L1 error of the
   rates drops from 23 % to 3 % at nsub = 8 (photon, c = 10, gam = 0.15, nk = 12).
5. s window: the frozen states (n_alpha = 0, n_beta = 1) no longer exchange particles with the
   window. The old gain / loss terms from the frozen outside do not conserve the particle number
   (the filling projection compensated for that). Now the window is a closed system with its own
   exact filling. Not compared numerically with the old treatment. Measured error of d / d0
   against the full IBZ (nk = 24): below 1e-6 to 2e-3, of the order of the largest occupation left outside.
6. Stopping: `resid` instead of max |dn|. An absolute tolerance on the current scales with the bath
   amplitudes and does not see the slow pair number mode (last point of the validation list).
7. `solve_fixed` iterates mu, not n (old `solve_fixed` and `solve_anderson` merged).
   The start is given as potentials (`mu=`) or defaults to the Fermi state at the coldest bath.
8. Defaults: `qd = sqrt(4 pi)` for phonons, no cutoff for photons, `nsub = 8`.
9. Notes: eq. 235 (v = sgn(Delta) ...) and eq. 35, 207 (u v = -Delta / 2E) use opposite signs for v.
   The code follows eq. 235; only squares of the vertices enter, so the rates do not depend on it.
   Eq. 286 is written for ta = -tb; the code uses [2 (ta - tb) u v]^2, which reduces to it.

## Validation (all at small nk, seconds)

- Equilibrium against `ei_unified` (nk = 24, V = 4.4 and 7.8): Delta_0, m, Delta(T), T_c agree to 1e-10.
- Rates against old `ei_phonon` + `ei_ibz` (run through a numpy stand-in for jax, nk = 16):
  phonon vertex 5e-4, photon with z dipole 3e-3 (`box = 0`, `hh = 0.02`), cavity 1e-16.
- Detailed balance 1e-14; T1 = T2 gives the Fermi state to 1e-15 for all bath kinds.
- `solve` against `solve_fixed`: d / d0 agrees to 1e-6 over a T1 sweep (BCS and BEC), see validation.png.
- 18 hard points (T2 down to 0.001 T_c, hot normal states, BEC, gam down to 1e-4): 17 converge with
  err < 1e-9. The remaining one (gam = 1e-4 at nk = 16) returns `ok = False` after 0.3 s.
- Realistic cavity + phonon amplitudes (T1 = 1.5 T_c, T2 = 0.3 T_c, nk = 16): mixing without the pair
  correction still has a relative pair flux of order one after 3000 iterations and gives
  d / d0 = 0.9991 instead of 0.9649 (BEC), 0.9976 instead of 0.9934 (BCS).

## JAX status

jax is not installed in the sandbox where this was written, so `gpu/` has not been run under real
jax. It was generated from `cpu/` (same statements, `jnp` for `np`, `jax.jit` on the kernels,
`lax.fori_loop` for the loops, no in place writes, no Python branches on traced values) and executed
through a numpy stand-in, where it reproduces `cpu/` exactly. First thing to run on a machine with
jax: `python -m EI_ness.check`.

## Not ported

Time integration (`solve_open`), block / pair current modes, the full BZ mode, `eff_T`, energy bin
phonons, the s binned model, `g_mat` / `source_rates` for vertex plots, the old plotting helpers.

## Possible improvements

- gam: the cell average only covers the phonon momentum, not the electron energies inside a cell.
  `pack` warns when gam + cs dk < 2 t dk. A pair dependent width floor (|v_i| + |v_j|) dk / 2
  would remove this grid noise but needs rates without tables.
- S = R / thermal factor is symmetric: only 3 of the 4 band blocks are needed (25 % of the build).
- Save the line tables to disk (they are cached per process only).
- The stability test of `solve` uses the flow dz / dt = F with equal rates for log d and m.
