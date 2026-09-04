from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax
import numpy as np
from tqdm.auto import tqdm

import EI.ei_jax as ej
import EI.ei_unified as eu


@dataclass(frozen=True)
class EffSol:
    beta: float
    t: float
    d: float
    m: float
    n: np.ndarray
    st: eu.MFState
    cur: np.ndarray
    power: float
    var: float
    err: float
    it: int
    ok: bool
    hist: dict[str, np.ndarray]


def solve_eff(
    bd: eu.Bands,
    p: eu.MFPars,
    bs: Iterable[ej.Bath],
    beta0: float,
    d: float = 0.2,
    m: float = 0.0,
    mu: float | None = None,
    dt: float = 0.05,
    mix: tuple[float, float] = (0.15, 0.15),
    tol: float = 1.0e-6,
    nmax: int = 20000,
    chk: int = 10,
    fac: float = 0.2,
    mode: str = "block",
    block: int = 512,
    mf: bool = True,
    prog: bool = True,
) -> EffSol:
    """Find the thermal strong-scattering fixed point of two baths."""
    if beta0 <= 0.0 or dt <= 0.0 or tol <= 0.0:
        raise ValueError("beta0, dt, and tol must be positive")
    if nmax <= 0 or chk <= 0 or block <= 0:
        raise ValueError("nmax, chk, and block must be positive")
    if fac <= 0.0 or fac >= 1.0:
        raise ValueError("fac must lie between zero and one")

    ad, am = map(float, mix)
    if ad < 0.0 or am < 0.0:
        raise ValueError("mix values must be nonnegative")

    bs = tuple(bs)
    if not bs:
        raise ValueError("at least one bath is required")

    mu = 0.5 * p.h * p.n if mu is None else float(mu)
    be = float(beta0)
    d = float(d)
    m = float(m)
    tt = 0.0
    ok = False

    hs = {q: [] for q in (
        "it", "t", "beta", "temp", "power", "var", "db",
        "d", "m", "n0", "nrate", "err",
    )}

    bar = tqdm(range(1, nmax + 1), desc="Effective temperature", disable=not prog)

    for it in bar:
        st = ej.mf_state(bd, p, d, m)
        x = np.asarray(jax.device_get(st.e), dtype=float) - mu
        z = np.clip(be * x, -500.0, 500.0)
        n = np.asarray(1.0 / (np.exp(z) + 1.0), dtype=np.float32)

        dn = ej.total_current(bd, st, n, bs, mode=mode, block=block)
        dn = np.asarray(jax.device_get(dn), dtype=float)

        power = float(np.sum(bd.w * np.sum(x * dn, axis=0)))
        var = float(np.sum(bd.w * np.sum(x * x * n * (1.0 - n), axis=0)))
        nr = float(np.sum(bd.w * np.sum(dn, axis=0)))
        nf = float(np.sum(bd.w * np.sum(n, axis=0)))

        if var <= np.finfo(float).tiny:
            if abs(power) < tol:
                db = 0.0
            else:
                raise FloatingPointError("thermal energy variance vanished")
        else:
            db = -power / var

        tg = ej.targets(bd, p, st, n)
        td, tm = jax.device_get((tg.d, tg.m))
        rd = float(td) - d
        rm = float(tm) - m
        er = max(abs(db), abs(nf - p.n))
        if mf:
            er = max(er, abs(rd), abs(rm))

        if it == 1 or it % chk == 0 or er < tol:
            hs["it"].append(it)
            hs["t"].append(tt)
            hs["beta"].append(be)
            hs["temp"].append(1.0 / be)
            hs["power"].append(power)
            hs["var"].append(var)
            hs["db"].append(db)
            hs["d"].append(d)
            hs["m"].append(m)
            hs["n0"].append(nf)
            hs["nrate"].append(nr)
            hs["err"].append(er)
            if prog:
                bar.set_postfix(err=f"{er:.2e}", T=f"{1.0 / be:.5f}")

        if er < tol:
            ok = True
            break

        h = min(dt, fac * be / max(abs(db), np.finfo(float).tiny))
        be = max(be + h * db, np.finfo(float).tiny)
        if mf:
            d += ad * rd
            m += am * rm
        tt += h

    stj = ej.mf_state(bd, p, d, m)
    x = np.asarray(jax.device_get(stj.e), dtype=float) - mu
    z = np.clip(be * x, -500.0, 500.0)
    n = np.asarray(1.0 / (np.exp(z) + 1.0), dtype=np.float32)
    dn = ej.total_current(bd, stj, n, bs, mode=mode, block=block)
    dn = np.asarray(jax.device_get(dn), dtype=float)
    power = float(np.sum(bd.w * np.sum(x * dn, axis=0)))
    var = float(np.sum(bd.w * np.sum(x * x * n * (1.0 - n), axis=0)))

    if var <= np.finfo(float).tiny:
        db = 0.0 if abs(power) < tol else np.inf
    else:
        db = -power / var

    nf = float(np.sum(bd.w * np.sum(n, axis=0)))
    tg = ej.targets(bd, p, stj, n)
    td, tm = jax.device_get((tg.d, tg.m))
    er = max(abs(db), abs(nf - p.n))
    if mf:
        er = max(er, abs(float(td) - d), abs(float(tm) - m))

    st = eu.mf_state(bd, p, d, m)
    hh = {q: np.asarray(v) for q, v in hs.items()}
    return EffSol(
        be, 1.0 / be, d, m, n, st, dn, power, var,
        float(er), it, ok or er < tol, hh,
    )


__all__ = ["EffSol", "solve_eff"]
