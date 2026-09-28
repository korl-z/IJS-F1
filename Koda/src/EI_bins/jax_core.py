"""Compiled double precision numerical core for the shell Newton solver."""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import sigmoid

from .kernel import delta_brk, delta_geo, spec_brk, spec_geo

jax.config.update("jax_enable_x64", True)

TINY = 1e-300


def make_core(pb):
    sh, bd, mf = pb.sh, pb.band, pb.mf
    ss = jnp.asarray(sh.ss)
    ws = jnp.asarray(sh.ws)
    w = jnp.asarray(sh.w)
    wt = jnp.asarray(pb.wt)
    cs = tuple(jnp.asarray(b.cm) for b in pb.baths)
    amps = tuple(2.0 * np.pi * b.amp for b in pb.baths)
    # baths with the same kernel and line width share the geometric part
    kg = {}
    for i, (b, kn) in enumerate(zip(pb.baths, pb.A)):
        kg.setdefault((id(kn), b.gam), (kn, b.gam, []))[2].append(i)
    bdat = tuple((jnp.tile(jnp.asarray(kn.A), (1, 2, 2)), jnp.asarray(kn.om),
                  float(kn.dw), float(g), tuple(ix)) for kn, g, ix in kg.values())
    N, nc, nth = pb.wt.size, pb.nc, pb.nth
    normal = pb.normal
    eye = jnp.eye(N)
    ids = jnp.arange(N)
    groups = tuple(tuple(int(c) for c in g) for g in pb.grp)
    bm = jnp.asarray(pb.blk)
    masks = tuple((jnp.asarray(pb.blk == c),
                   jnp.asarray(np.isin(pb.blk, g) & (pb.blk != c)))
                  for g in pb.grp for c in g)
    labels = tuple(c for g in pb.grp for c in g)

    def cl_calc(d, m):
        ea = 0.5 * bd.gap - 2.0 * bd.ta * ss
        eb = -0.5 * bd.gap - 2.0 * bd.tb * ss
        xi = 0.5 * (ea - eb) - 0.5 * mf.h * m
        et = 0.5 * (ea + eb) + 0.5 * mf.h * (mf.n - 1.0)
        ek = jnp.hypot(xi, d)
        es = jnp.stack((et + ek, et - ek))
        e = jnp.sum(ws * es, axis=-1)
        ns2 = ss.shape[1] ** 2
        cor = 12.0 * ns2 / max(ns2 - 1.0, 1.0)
        hw = jnp.sqrt(jnp.maximum(cor * jnp.sum(ws * (es - e[..., None]) ** 2,
                                                 axis=-1), 1e-24))
        ok = ek > 0.0
        ep = jnp.where(ok, ek, 1.0)
        xr = jnp.where(ok, xi / ep, 1.0)
        uv = jnp.where(ok, d / (2.0 * ep), 0.0)
        avg = lambda x: jnp.sum(ws * x, axis=-1)
        a, b, c = avg(0.5 * (1.0 + xr)), avg(0.5 * (1.0 - xr)), avg(uv)
        ma = jnp.stack((jnp.stack((a, -c), -1), jnp.stack((-c, b), -1)), -2)
        mb = jnp.stack((jnp.stack((b, c), -1), jnp.stack((c, a), -1)), -2)
        ie = jnp.where(ok, 0.5 / ep, 0.0)
        return e, hw, ma, mb, es, uv, xr, ie

    def rate(cl, ts):
        e, hw, ma, mb = cl[:4]
        e, hw = e.reshape(-1), hw.reshape(-1)
        x = e[None, :] - e[:, None]
        ha, hb = hw[:, None], hw[None, :]
        cf = jnp.max(hw)
        mats = (ma, mb)
        # frequency bin sums of A_m times the line bracket, one per bath
        pp = [None] * len(amps)
        for A3, om, dw, gam, ix in bdat:
            cmin = jnp.maximum(cf, dw)

            def body(k, acc, A3=A3, om=om, gam=gam, ix=ix, cmin=cmin):
                if gam > 0.0:
                    geo = spec_geo(jnp, x, ha, hb, cmin, om[k], gam)
                    brs = [spec_brk(jnp, geo, ts[i]) for i in ix]
                else:
                    le = delta_geo(jnp, x, ha, hb, cmin, om[k])
                    brs = [delta_brk(jnp, le, om[k], ts[i]) for i in ix]
                return tuple(a + A3[k] * br for a, br in zip(acc, brs))

            acc = lax.fori_loop(0, om.shape[0], body,
                                tuple(jnp.zeros_like(x) for _ in ix))
            for i, a in zip(ix, acc):
                pp[i] = a
        K = jnp.zeros_like(x)
        for i, (cm, amp) in enumerate(zip(cs, amps)):
            blocks = []
            for mi in mats:
                a = jnp.einsum('ab,iac,cd->ibd', cm, mi, cm)
                blocks.append(jnp.concatenate(
                    [jnp.einsum('ibd,jbd->ij', a, mj) for mj in mats], axis=1))
            p = amp * jnp.concatenate(blocks, axis=0) * pp[i]
            t = ts[i]
            up = jnp.where(t > 0.0, p.T * jnp.exp(jnp.minimum(x, 0.0) /
                                                  jnp.maximum(t, 1e-300)), 0.0)
            K = K + jnp.where(x >= 0.0, p, up)
        return K.at[ids, ids].set(0.0)

    def coef(cl):
        e, es = cl[0], cl[4]
        dp = jnp.pad(e[:, 1:] - e[:, :-1], ((0, 0), (0, 1)))
        dm = jnp.pad(e[:, :-1] - e[:, 1:], ((0, 0), (1, 0)))
        s2 = dp**2 + dm**2
        s2 = jnp.where(s2 > 0.0, s2, 1.0)
        de = es - e[..., None]
        cp = de * (dp / s2)[..., None]
        cm = de * (dm / s2)[..., None]
        return 1.0 - cp - cm, cm, cp

    def sub(mu, co):
        mu = mu.reshape(2, -1)
        mp = jnp.pad(mu[:, 1:], ((0, 0), (0, 1)))
        mm = jnp.pad(mu[:, :-1], ((0, 0), (1, 0)))
        c0, cm, cp = co
        return c0 * mu[..., None] + cm * mm[..., None] + cp * mp[..., None]

    def grad(g, co):
        c0, cm, cp = co
        a = jnp.sum(g * c0, axis=-1)
        a += jnp.pad(jnp.sum(g * cm, axis=-1)[:, 1:], ((0, 0), (0, 1)))
        a += jnp.pad(jnp.sum(g * cp, axis=-1)[:, :-1], ((0, 0), (1, 0)))
        return a.reshape(-1)

    def state(z, ts):
        mu = z[:N]
        d = 0.0 if normal else z[N]
        m = z[N + nth - 1]
        cl = cl_calc(d, m)
        K = rate(cl, ts)
        n, p = sigmoid(-mu), sigmoid(mu)
        return cl, K, n, p, K @ n, K.T @ p

    def fields(cl, mu, deriv=False):
        mus = sub(mu, coef(cl))
        nsb = sigmoid(-mus)
        ww = w[:, None] * ws
        dn = nsb[1] - nsb[0]
        rd = 1.0 - mf.v * jnp.sum(ww * cl[7] * dn)
        rm = -jnp.sum(ww * cl[6] * dn)
        if not deriv:
            return rd, rm
        dns = -nsb * sigmoid(mus)
        gd = jnp.stack((ww * cl[7], -ww * cl[7])) * (mf.v * dns)
        gm = jnp.stack((-ww * cl[6], ww * cl[6])) * dns
        return rd, rm, grad(gd, coef(cl)), grad(gm, coef(cl))

    def f(z, ts, tgt, act, blk, form, mu0):
        mu = z[:N]
        cl, K, n, p, a, b = state(z, ts)
        ks = (jnp.where(act, z[N + nth + jnp.maximum(blk, 0)], 0.0)
              if nc else jnp.zeros(N))
        r = jnp.where(act, mu + jnp.log(jnp.maximum(a, TINY)) -
                      jnp.log(jnp.maximum(b, TINY)) + ks, mu - mu0)
        rd, rm = fields(cl, mu)
        sn = jnp.bincount(jnp.maximum(blk, 0), weights=wt * n * act, length=nc)
        sp = jnp.bincount(jnp.maximum(blk, 0), weights=wt * p * act, length=nc)
        fill = jnp.where(form, jnp.log(jnp.maximum(sn, TINY)),
                         jnp.log(jnp.maximum(sp, TINY))) - jnp.log(tgt)
        mid = jnp.array([z[N] - rm]) if normal else jnp.stack((rd, z[N + 1] - rm))
        return jnp.concatenate((r, mid, fill))

    def fj(z, ts, tgt, act, blk, form, mu0):
        cl, K, n, p, a, b = state(z, ts)
        F = f_from_state(z, cl, K, n, p, a, b, tgt, act, blk, form, mu0)
        dn = -n * p
        bal = eye + (K / jnp.maximum(a, TINY)[:, None] +
                     K.T / jnp.maximum(b, TINY)[:, None]) * dn[None, :]
        bal = jnp.where(act[:, None], bal, eye)
        sl = (blk[:, None] == jnp.arange(nc)[None, :]) & act[:, None]
        _, _, gd, gm = fields(cl, z[:N], True)
        mid = gm[None, :] if normal else jnp.stack((gd, gm))
        sn = jnp.bincount(jnp.maximum(blk, 0), weights=wt * n * act, length=nc)
        sp = jnp.bincount(jnp.maximum(blk, 0), weights=wt * p * act, length=nc)
        row = jnp.where(form[:, None], wt[None, :] * dn[None, :] /
                        sn[:, None], -wt[None, :] * dn[None, :] / sp[:, None])
        fill = jnp.where((blk[None, :] == jnp.arange(nc)[:, None]) & act[None, :],
                         row, 0.0)
        left = jnp.concatenate((bal, mid, fill), axis=0)
        right = jnp.concatenate((sl.astype(jnp.float64),
                                 jnp.zeros((nth + nc, nc))), axis=0)
        J = jnp.concatenate((left, jnp.zeros((N + nth + nc, nth)), right), axis=1)
        def col(j):
            k = N + j
            h = 1e-7 * jnp.maximum(jnp.abs(z[k]), 0.1)
            zz = jnp.zeros_like(z).at[k].set(h)
            return (f(z + zz, ts, tgt, act, blk, form, mu0) -
                    f(z - zz, ts, tgt, act, blk, form, mu0)) / (2.0 * h)
        cols = jax.vmap(col)(jnp.arange(nth)).T
        return F, J.at[:, N:N + nth].set(cols)

    def f_from_state(z, cl, K, n, p, a, b, tgt, act, blk, form, mu0):
        mu = z[:N]
        ks = (jnp.where(act, z[N + nth + jnp.maximum(blk, 0)], 0.0)
              if nc else jnp.zeros(N))
        r = jnp.where(act, mu + jnp.log(jnp.maximum(a, TINY)) -
                      jnp.log(jnp.maximum(b, TINY)) + ks, mu - mu0)
        rd, rm = fields(cl, mu)
        sn = jnp.bincount(jnp.maximum(blk, 0), weights=wt * n * act, length=nc)
        sp = jnp.bincount(jnp.maximum(blk, 0), weights=wt * p * act, length=nc)
        fill = jnp.where(form, jnp.log(jnp.maximum(sn, TINY)),
                         jnp.log(jnp.maximum(sp, TINY))) - jnp.log(tgt)
        mid = jnp.array([z[N] - rm]) if normal else jnp.stack((rd, z[N + 1] - rm))
        return jnp.concatenate((r, mid, fill))

    def err(z, F):
        return jnp.max(jnp.where(jnp.arange(F.size) < N,
                                  jnp.where(jnp.abs(z[jnp.minimum(jnp.arange(F.size), N-1)])
                                            <= 50.0, jnp.abs(F), 0.0), jnp.abs(F)))

    def step(z, ts, tgt, act, blk, form, mu0):
        args = ts, tgt, act, blk, form, mu0
        F, J = fj(z, *args)
        dz = jnp.linalg.solve(J, -F)
        dz = dz.at[:N].set(jnp.clip(dz[:N], -40.0, 40.0))
        s = jnp.where((not normal) & (dz[N] < -0.5 * z[N]),
                      jnp.minimum(1.0, -0.5 * z[N] / dz[N]), 1.0)
        wr = jnp.ones_like(z)
        wp = jnp.sqrt(sigmoid(-z[:N]) * sigmoid(z[:N]))
        wr = wr.at[:N].set(wp / jnp.maximum(jnp.max(wp), TINY))
        f0 = jnp.sum((wr * F)**2)
        def cond(c):
            s1, _, _, good = c
            return (s1 > 1e-6) & ~good
        def body(c):
            s1, _, _, _ = c
            z1 = z + s1 * dz
            F1 = f(z1, *args)
            good = jnp.all(jnp.isfinite(F1)) & (jnp.sum((wr * F1)**2) <
                                                 (1.0 - 1e-4 * s1) * f0)
            return (jnp.where(good, s1, s1 * 0.5), z1, F1, good)
        s1, z1, F1, good = lax.while_loop(cond, body, (s, z, F, jnp.array(False)))
        return jnp.where(good, z1, z), s1, err(z, F), jnp.all(jnp.isfinite(dz))

    def flux(z, ts):
        cl, K, n, p, a, b = state(z, ts)
        out = jnp.zeros(nc)
        for c, (ic, ot) in zip(labels, masks):
            fin = (p * ic) @ (K @ (n * ot))
            fout = (n * ic) @ (K.T @ (p * ot))
            out = out.at[c].set(0.5 * (jnp.log(jnp.maximum(fin, TINY)) -
                                       jnp.log(jnp.maximum(fout, TINY))))
        return out

    return tuple(jax.jit(g) for g in (f, fj, step, flux))
