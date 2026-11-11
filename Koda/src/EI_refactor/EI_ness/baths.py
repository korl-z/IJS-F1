"""Bath definitions and their static data. Numpy only, shared by the cpu and gpu solvers.

Toy:     factorizable toy rates.
Phonon:  phonons on the grid, Peierls vertex (eq. 97 - 99 with D = 0).
Photon:  cavity photons on the grid, TE1 + TM1 and the z dipole TM2 (eq. 258).
Cavity:  photons with q integrated analytically, vertical transitions (eq. 286).

Lines are always Lorentzians (eq. 272), gam > 0. Grid baths conserve momentum exactly
and use the cell averaged line of eq. 274, 275.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

QD = float(np.sqrt(4.0 * np.pi))  # Debye radius, pi qd^2 = BZ area
ALPHA = 1.0 / 137.036


@dataclass(frozen=True)
class Toy:
    """R_ij = kap[band_i, band_j] exp(-(dE / wc)^2) f(E_i, E_j) (q_i q_j)^2."""

    t: float
    kap: Any = 1.0  # scalar, (aa, ab, bb) or 2 x 2, [dest band, src band]
    wc: float = np.inf
    orb: str | None = None  # None: q = 1, "a": q = (u, v), "b": q = (-v, u)
    mu: float = 0.0  # in the shifted energy frame
    rate: str = "product"  # product: f(E_i) (1 - f(E_j)), db: 1 / (1 + e^(dE / t))


@dataclass(frozen=True)
class Phonon:
    t: float
    cs: float
    gam: float
    amp: float  # 1 / (2 M)
    tpa: float  # Peierls slope t'_a
    tpb: float
    w0: float = 0.0  # omega_q = hypot(w0, cs q)
    qd: float = QD
    nsub: int = 8  # tent nodes per direction in the cell average
    box: float = 1.0  # 1: node lines averaged over their om interval, 0: plain sum of eq. 275


@dataclass(frozen=True)
class Photon:
    t: float
    c: float
    w0: float
    gam: float
    amp: float | None = None  # None: 4 alpha w0 (eq. 280)
    dip: float = 0.0  # z dipole
    qd: float = np.inf
    nsub: int = 8
    box: float = 1.0


@dataclass(frozen=True)
class Cavity:
    t: float
    w0: float
    gam: float
    c: float = 3945.0
    amp: float | None = None  # None: 4 alpha w0


@dataclass(frozen=True)
class Chord:
    """Phonons on the s grid (model.sgrid): the momentum sum is done once in stab."""

    t: float
    cs: float
    gam: float  # 0: sharp line, integrated exactly (needs w0 = 0)
    amp: float  # 1 / (2 M)
    tpa: float
    tpb: float
    w0: float = 0.0
    qd: float = QD
    nq: int = 128
    nphi: int = 64


class Pack(NamedTuple):
    toy: Any  # rows (t, wc, orb, mu, prod, kaa, kab, kba, kbb), or None
    lin: Any  # per grid bath: (par = (t, amp, gam, c1a, c1b, dip), tab, cen)
    cav: Any  # rows (t, w0, gam, amp, c), or None
    hh: float  # step of the line tables
    tmin: float  # coldest bath
    cho: Any = ()  # per chord bath: (par = (t, amp, gam, tpa, tpb, cs, w0, qd), S, dq)


KINDS = {"toy": Toy, "phonon": Phonon, "photon": Photon, "cavity": Cavity, "chord": Chord}


def make(spec, ts=None, tu=1.0):
    """Baths from dicts {kind: ..., t: ..., ...}; ts overrides the temperatures, tu is their unit."""
    out = []
    for i, s in enumerate(spec):
        s = dict(s)
        cls = KINDS[s.pop("kind")]
        s["t"] = tu * (s["t"] if ts is None else ts[i])
        out.append(cls(**s))
    return out


def load(path):
    """yaml file -> (Model, Grid, bath spec, full dict). cfg['use'] picks the bath set."""
    import yaml
    from .model import Model, grid
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Model(**cfg["model"]), grid(**cfg["grid"]), cfg["baths"][cfg["use"]], cfg


def em_bound(md, mar=0.1):
    """Bound on |E_i - E_j| for all d, m (|d| <= V / 2, |m| <= min(n, 2 - n)), plus a margin."""
    xm = abs(md.gap) / 2 + 2 * abs(md.ta - md.tb) + abs(md.h) * min(md.n, 2 - md.n) / 2
    return (1 + mar) * (4 * abs(md.ta + md.tb) + 2 * np.hypot(xm, abs(md.v) / 2))


def _kap(kap):
    a = np.asarray(kap, dtype=float)
    if a.ndim == 0:
        a = np.full((2, 2), float(a))
    elif a.shape == (3,):
        a = np.array([[a[0], a[1]], [a[1], a[2]]])
    if a.shape != (2, 2) or np.any(~np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("kap must be a nonnegative scalar, length 3, or 2 x 2")
    return a


@lru_cache(maxsize=16)
def ltab(nk, cs, w0, gam, qd, nsub, box, phn, ymax, hh):
    """Cell averaged lines (eq. 274, 275) for every grid transfer q.

    Energy grid per q: w = cen_q + gam sinh(hh (i - M)), fine at the line and coarse
    in the tails, so the size grows only like log(1 / gam).
    Rows, phonons: F. Photons: Zxx F, Zyy F, Zxy F at w_q and the TM2 factor at w_2q.
    F = 8 gam / (((w - om)^2 + gam^2) ((w + om)^2 + gam^2)) (eq. 272), averaged for every
    tent node over the om interval that the node stands for (box = 1; box = 0: point value,
    the plain sum of eq. 275). The node sum then stays smooth for any gam > 0.
    Returns tab (rows, nk^2, 2 M + 1) and the centres cen (rows, nk^2).
    """
    dk = 2.0 * np.pi / nk

    def wr(q):
        return np.mod(q + np.pi, 2.0 * np.pi) - np.pi

    q0 = wr(dk * np.arange(nk))
    qq = np.hypot(q0[:, None], q0[None, :])
    mm = int(np.ceil(np.arcsinh(ymax / gam) / hh))
    y = gam * np.sinh(hh * (np.arange(2 * mm + 1) - mm))
    cen = np.stack([np.hypot(f * w0, cs * qq) for f in ((1.0,) if phn else (1.0, 1.0, 1.0, 2.0))])
    t = 2.0 * ((np.arange(nsub) + 0.5) / nsub - 0.5)  # tent nodes in [-1, 1]
    wt = 1.0 - np.abs(t)
    wt /= wt.sum()
    z = np.zeros(cen.shape + y.shape)

    def line(w, om, st):
        """Eq. 272 averaged over the om box of one node (width dl); dl -> 0 gives eq. 272."""
        dl = box * st / np.maximum(om, 1.0e-300) / gam
        c1, c2 = ((1.0 + (((w + sg * om[..., None]) / gam) ** 2 - 0.25 * dl[..., None] ** 2))
                  for sg in (-1.0, 1.0))
        num = 4.0 * dl[..., None] * w * om[..., None] / gam**2
        den = c1 * c2 + dl[..., None] ** 2
        lin = np.abs(num) > 1.0e-8 * den
        return 8.0 / gam**3 * np.where(lin, np.arctan2(num, den) / np.where(lin, num, 1.0), 1.0 / den)

    for tx, ax in zip(t, wt):
        for ty, ay in zip(t, wt):
            qx, qy = wr(q0 + tx * dk)[:, None], wr(q0 + ty * dk)[None, :]
            qm = np.hypot(qx, qy)
            ok = ((qm <= qd) * ax * ay)[..., None]
            st = cs**2 * np.maximum(np.abs(qx), np.abs(qy)) * 2.0 * dk / nsub  # om * (node step of om)
            om = np.hypot(w0, cs * qm)
            ln = ok * line(cen[0][..., None] + y, om, st)
            if phn:
                z[0] += ln
                continue
            ph = np.arctan2(qy, qx)
            cp, sn, r2 = np.cos(ph)[..., None], np.sin(ph)[..., None], ((w0 / om) ** 2)[..., None]
            z[0] += (sn * sn + r2 * cp * cp) * ln
            z[1] += (cp * cp + r2 * sn * sn) * ln
            z[2] += 2.0 * sn * cp * (r2 - 1.0) * ln
            o2 = np.hypot(2.0 * w0, cs * qm)
            z[3] += ok * ((cs * qm / o2) ** 2)[..., None] * line(cen[3][..., None] + y, o2, st)
    z, cen = z.reshape(len(cen), nk * nk, -1), cen.reshape(len(cen), -1)
    z.setflags(write=False)
    cen.setflags(write=False)
    return z, cen


@lru_cache(maxsize=8)
def stab(ns, nf, nq, qmax, nphi):
    """Pair table S[I, J, l] = <|sin(k + q) - sin k|^2 1[k in I] 1[k + q in J]> at |q| = l dq.

    The average runs over k on the nf x nf grid and over nphi directions of q.
    It holds all the momentum conservation; it does not depend on d, m, T or gam.
    """
    k = 2.0 * np.pi * np.arange(nf) / nf
    kx, ky = (q.ravel() for q in np.meshgrid(k, k, indexing="ij"))
    sb = np.linspace(-2.0, 2.0, ns + 1)
    bi = np.clip(np.digitize(np.cos(kx) + np.cos(ky), sb) - 1, 0, ns - 1)
    dq = qmax / (nq - 1)
    S = np.zeros((nq, ns * ns))
    for l in range(1, nq):
        for ph in 2.0 * np.pi * (np.arange(nphi) + 0.5) / nphi:
            x, y = kx + l * dq * np.cos(ph), ky + l * dq * np.sin(ph)
            v2 = (np.sin(x) - np.sin(kx)) ** 2 + (np.sin(y) - np.sin(ky)) ** 2
            bj = np.clip(np.digitize(np.cos(x) + np.cos(y), sb) - 1, 0, ns - 1)
            S[l] += np.bincount(bi * ns + bj, weights=v2, minlength=ns * ns)
    S = S.reshape(nq, ns, ns) / (nphi * kx.size)
    S = np.moveaxis(0.5 * (S + S.transpose(0, 2, 1)), 0, -1)
    S.setflags(write=False)
    return S, dq


def pack(bs, md, g, hh=0.05):
    """Static data of the baths on the grid g, as plain arrays."""
    nk, em = int(g.nk), em_bound(md)
    toy, lin, cav, cho = [], [], [], []
    for b in bs:
        if isinstance(b, (Phonon, Photon)) != (g.ki is not None) and not isinstance(b, (Toy, Cavity)):
            raise ValueError("Phonon and Photon need model.grid, Chord needs model.sgrid")
        if isinstance(b, Toy):
            if b.rate not in ("product", "db") or b.orb not in (None, "a", "b"):
                raise ValueError("toy rate must be product or db, orb None, a or b")
            toy.append((b.t, float(b.wc), {None: 0.0, "a": 1.0, "b": 2.0}[b.orb], b.mu,
                        float(b.rate == "product"), *_kap(b.kap).reshape(-1)))
        elif isinstance(b, Chord):
            S, dq = stab(g.s.size, nk, int(b.nq), float(b.qd), int(b.nphi))
            if b.gam == 0.0 and b.w0 != 0.0 or 0.0 < b.gam < b.cs * dq:
                raise ValueError("chord: gam = 0 needs w0 = 0, gam > 0 needs gam >= cs dq (raise nq)")
            cho.append((np.array((b.t, b.amp, b.gam, b.tpa, b.tpb, b.cs, b.w0, b.qd)), S, dq))
        elif isinstance(b, Cavity):
            cav.append((b.t, b.w0, b.gam, 4.0 * ALPHA * b.w0 if b.amp is None else b.amp, b.c))
        else:
            if b.gam <= 0.0:
                raise ValueError("gam must be positive: the lines are always Lorentzians")
            phn = isinstance(b, Phonon)
            cs = b.cs if phn else b.c
            if not phn and b.w0 <= 0.0:
                raise ValueError("the photon needs w0 > 0")
            if cs * 2.0 * np.pi / nk + b.gam < 4.0 * max(abs(md.ta), abs(md.tb)) * np.pi / nk:
                warnings.warn("gam + cs dk is below the electronic level spacing 2 t dk: "
                              "the rates of this bath are limited by grid noise", stacklevel=2)
            tab, cen = ltab(nk, float(cs), float(b.w0), float(b.gam), float(b.qd), int(b.nsub),
                            float(b.box), phn, float(em + np.hypot(2.0 * b.w0, cs * np.pi * 2**0.5)),
                            float(hh))
            amp = 4.0 * ALPHA * b.w0 if not phn and b.amp is None else b.amp
            c1 = (b.tpa, b.tpb, 0.0) if phn else (md.ta, md.tb, b.dip)
            lin.append((np.array((b.t, amp, b.gam, *c1), dtype=float), tab, cen))
        if b.t < 0.0:
            raise ValueError("t must be nonnegative")
    arr = lambda x: np.array(x, dtype=float) if x else None
    return Pack(arr(toy), tuple(lin), arr(cav), hh, float(min(b.t for b in bs)), tuple(cho))
