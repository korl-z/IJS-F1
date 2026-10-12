"""Logarithmic fixed point with Anderson acceleration.

Place alongside ei_jax_2.py. Rates are frozen within each checkpoint chunk.
Phonon rates are constructed in source-column blocks and kept as one matrix.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from EI import ei_jax_2 as ej


@partial(jax.jit, static_argnames=("block",), donate_argnums=(0,))
def _rates_blk(r, st, bp, block):
    """Overwrite the donated matrix using bounded source-column blocks."""
    nm = st.e.size
    block = min(block, nm)
    nb = (nm + block - 1) // block
    jj = jnp.arange(block, dtype=ej.I32)

    def body(i, rr):
        # Shift the last block inward so every source index remains valid.
        j0 = jnp.minimum(i * block, nm - block)
        rb = ej.pb.rates(st, bp, j=jj + j0)
        return jax.lax.dynamic_update_slice(rr, rb, (0, j0))

    return jax.lax.fori_loop(0, nb, body, r)


def _rates(st, bp, rf, block, r=None):
    """Build phonon rates in blocks; preserve the other bath backends."""
    if not isinstance(bp, ej.pb.Pack):
        return ej._rates_jit(st, bp, rf)
    if r is None:
        r = jnp.zeros((st.e.size, st.e.size), dtype=st.e.dtype)
    return _rates_blk(r, st, bp, block=block)


@jax.jit
def _proj(z, w, pn):
    mu = z[:-2]
    ww = jnp.tile(w, 2)
    h = jnp.max(jnp.abs(mu)) + 80.0

    def body(i, ab):
        lo, hi = ab
        s = 0.5 * (lo + hi)
        nf = jnp.sum(ww * jax.nn.sigmoid(-mu - s))
        return jnp.where(nf > pn, s, lo), jnp.where(nf > pn, hi, s)

    lo, hi = jax.lax.fori_loop(0, 64, body, (-h, h))
    return z.at[:-2].add(0.5 * (lo + hi))


@partial(jax.jit, static_argnames=("ns", "mem"))
def _chunk(ea, eb, w, pv, pn, ph, r, z, mix, sc, reg, cap, ns, mem):
    nm = 2 * w.size

    def ev(x):
        mu, d, m = x[:-2] * sc[:-2], x[-2] * sc[-2], x[-1] * sc[-1]
        n = jax.nn.sigmoid(-mu).reshape(2, -1)
        h = jax.nn.sigmoid(mu).reshape(2, -1)
        ww = jnp.tile(w, 2)
        a = (r @ (ww * n.reshape(-1))).reshape(n.shape)
        b = (r.T @ (ww * h.reshape(-1))).reshape(n.shape)
        good = jnp.all(a > 0.0) & jnp.all(b > 0.0)
        tg = ej._targets(ej._mf(ea, eb, pn, ph, d, m), n, w, pv)
        mt = jnp.log(b) - jnp.log(a)
        g = _proj(jnp.concatenate((mt.reshape(-1), jnp.stack((tg.d, tg.m)))), w, pn) / sc
        f = g - x
        er = jnp.max(jnp.abs(f))
        er = jnp.where(good & jnp.all(jnp.isfinite(f)), er, jnp.inf)
        return f, er

    def proj(x):
        return _proj(x * sc, w, pn) / sc

    def step(ca, _):
        x, xp, fp, dx, df, first, na = ca
        f, er = ev(x)
        sx = jnp.where(first, jnp.zeros_like(x), x - xp)
        sf = jnp.where(first, jnp.zeros_like(f), f - fp)
        dx = jnp.concatenate((dx[1:], sx[None, :]))
        df = jnp.concatenate((df[1:], sf[None, :]))
        q = df @ df.T
        lam = reg * jnp.maximum(jnp.trace(q) / mem, 1.0e-30)
        c = jnp.linalg.solve(q + lam * jnp.eye(mem), df @ f)
        xb = proj(x + mix * f)
        xa = x + mix * f - c @ (dx + mix[None, :] * df)
        dv = xa - x
        fac = jnp.minimum(1.0, cap / jnp.maximum(jnp.max(jnp.abs(dv[:nm])), 1.0e-30))
        xa = proj(x + fac * dv)
        _, ea0 = ev(xa)
        _, eb0 = ev(xb)
        # Compare extrapolation with the ordinary mixed step at frozen rates.
        acc = (~first) & jnp.isfinite(ea0) & (ea0 <= eb0) & (ea0 < er)
        xn = jnp.where(acc, xa, xb)
        # Restart after a rejected extrapolation.
        dx = jnp.where(acc | first, dx, jnp.zeros_like(dx))
        df = jnp.where(acc | first, df, jnp.zeros_like(df))
        return (xn, x, f, dx, df, jnp.asarray(False), na + acc), None

    x = z / sc
    zz = jnp.zeros((mem, z.size), dtype=z.dtype)
    ca = (x, x, jnp.zeros_like(x), zz, zz, jnp.asarray(True), jnp.asarray(0))
    ca, _ = jax.lax.scan(step, ca, xs=None, length=ns)
    return ca[0] * sc, ca[-1]


def solve_anderson(bd, p, n, bs, d=0.2, m=0.0, mix=(0.2, 0.2, 0.2),
                   tol=1.0e-6, nmax=5000, chk=10, mem=4, reg=1.0e-8,
                   cap=5.0, scale=None, prog=True, block=128):
    """Same physical stopping residual and return type as solve_fixed.

    Exact initial zeros and ones are moved inside the interval before taking
    logarithms. This is only an initialization adjustment; occupations are
    not clipped during iteration. Requires positive gain/loss for every state.
    Filling is imposed by a common shift of all GGE potentials.
    History restarts whenever rates are refreshed. Time is iteration count.
    block limits source columns per phonon rate-construction block. The full
    matrix is cached between checkpoints and its buffer is donated on refresh.
    Other bath types retain their existing rate builder.
    """
    n = ej._init_n(bd, p, n, d, m, tol, nmax, chk)
    if not 0.0 < p.n < 2.0:
        raise ValueError("finite potentials require filling strictly between 0 and 2")
    mx = np.asarray(mix, dtype=float)
    if mx.shape != (3,) or np.any(~np.isfinite(mx)) or np.any(mx <= 0) or np.any(mx > 1):
        raise ValueError("mix must contain three values in (0, 1]")
    if int(mem) != mem or mem < 1 or not np.isfinite(reg) or reg <= 0 or not np.isfinite(cap) or cap <= 0:
        raise ValueError("mem, reg and cap must be positive")
    if isinstance(block, (bool, np.bool_)) or not isinstance(block, (int, np.integer)) or block < 1:
        raise ValueError("block must be a positive integer")
    block = int(block)
    ea, eb, w, pv, pn, ph = ej._pars(bd, p)
    bp, rf = ej._baths(bs)
    nm = 2 * bd.size
    sd, sm = (max(abs(d), 0.1), max(abs(m), 0.1)) if scale is None else scale
    if not np.all(np.isfinite([sd, sm])) or min(sd, sm) <= 0:
        raise ValueError("scale must contain two positive finite values")
    sc = jnp.concatenate((jnp.ones(nm), jnp.asarray([sd, sm])))
    mx = jnp.concatenate((jnp.full(nm, mx[0]), jnp.asarray(mx[1:])))
    ep = 1.0e-14
    nc = jnp.clip(n, ep, 1.0 - ep)
    mu = jnp.log1p(-nc) - jnp.log(nc)
    z = _proj(jnp.concatenate((mu.reshape(-1), jnp.asarray([d, m]))), w, pn)
    r = _rates(ej._mf(ea, eb, pn, ph, z[-2], z[-1]), bp, rf, block)
    hs = {q: [] for q in ("it", "t", "err", "cur", "ed", "em", "d", "m", "n0", "aa")}
    ok, it = False, 0
    bar = tqdm(range(0, int(nmax), int(chk)), desc="Anderson EI JAX", disable=not prog)
    for i in bar:
        ns = min(int(chk), int(nmax) - i)
        z, na = _chunk(ea, eb, w, pv, pn, ph, r, z, mx, sc,
                       reg, cap, ns=ns, mem=int(mem))
        n = jax.nn.sigmoid(-z[:-2]).reshape(2, -1)
        d, m = z[-2], z[-1]
        r = _rates(ej._mf(ea, eb, pn, ph, d, m), bp, rf, block, r)
        dn, erj, ec, ed, em, nf = ej._fixed_diag(ea, eb, w, pv, pn, ph, r, n, d, m)
        vals = np.asarray(jax.device_get(jnp.stack((erj, ec, ed, em, d, m, nf, na))))
        if np.any(~np.isfinite(vals)):
            raise RuntimeError("nonfinite logarithmic iteration; check positive gain and loss")
        it = i + ns
        for q, v in zip(hs, (it, float(it), *vals)):
            hs[q].append(v)
        if prog:
            bar.set_postfix(err=f"{vals[0]:.2e}", d=f"{vals[4]:.5f}")
        if vals[0] < tol:
            ok = True
            break
    bar.close()
    return ej._result(bd, p, n, d, m, dn, erj, it, float(it), ok, hs)
