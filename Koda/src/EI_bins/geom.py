"""Bath parameters and the static momentum pair kernels (photon vertex).

|M~|^2 = (amp / w_q) [(X_TE^2 + r2 X_TM^2) C1^2 + (sin^2 phi + r2 cos^2 phi) C2^2]
for |q| <= qd (BZ wrapped q), r2 = (w0 / w_q)^2, X from the velocities
sin(p + q / 2) at the midpoint, dipole along x (eqs. 125, 128, 129). w_q is
constant, acoustic (cs |q|) or gapped (sqrt(w0^2 + cs^2 q^2)). The binned
kernels of one bath are the sums over k in I, p in J of w_k w_p times the
velocity factor (A) and the dipole factor (D), both with 1 / w_q, restricted
to w_{k-p} in bin m. They depend only on the shells and the bath geometry,
not on (T, d, m). The coherence factors C1, C2 are applied in kernel.rates.
pair_kernel is the original phonon kernel q^2 / (q^2 + lam^2)^2.
"""

import hashlib
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import numpy as np
import scipy.fft as sfft


@dataclass(frozen=True)
class Phonon:
    """Phonon bath coupled to the physical orbitals through c.

    w0 is the Einstein frequency ('constant') or the gap ('gapped'); it is not
    used for 'acoustic'. gam > 0 replaces the delta lines by damped
    oscillator spectral functions of width gam. eta > 0 is a fixed box
    width of the line (energy units), used instead of the grid cell width
    when larger, so that gam = 0 has a limit for ns -> infinity.
    """

    t: float
    amp: float = 1.0
    w0: float = 1.0
    lam: float = 1.0
    qd: float = np.pi * np.sqrt(2.0)
    c: tuple = ((1.0, 0.0), (0.0, 1.0))
    name: str = "phonon"
    disp: str = "constant"  # 'constant', 'acoustic' or 'gapped'
    cs: float = 1.0  # sound velocity, acoustic and gapped
    gam: float = 0.0  # line width; 0 means delta lines
    eta: float = 0.0  # fixed box width of the line, 0: one grid cell
    dip: float = 0.0  # dipole strength, along x

    def __post_init__(self):
        c = np.asarray(self.c, dtype=float)
        if c.shape != (2, 2) or not np.isfinite(c).all():
            raise ValueError("c must be a finite 2x2 matrix")
        if not np.allclose(c, c.T):
            raise ValueError("c must be symmetric, c_ab = c_ba")
        if self.t < 0 or self.amp < 0:
            raise ValueError("t and amp must be nonnegative")
        if min(self.lam, self.qd) <= 0 or min(self.w0, self.cs) < 0:
            raise ValueError("lam and qd must be positive, w0 and cs nonnegative")
        if self.disp not in ("constant", "acoustic", "gapped"):
            raise ValueError("disp must be 'constant', 'acoustic' or 'gapped'")
        wm = {"constant": self.w0, "acoustic": self.cs,
              "gapped": max(self.w0, self.cs)}[self.disp]
        if wm <= 0:
            raise ValueError("the dispersion needs a positive w0 or cs")
        if self.gam < 0 or not np.isfinite(self.gam):
            raise ValueError("gam must be finite and nonnegative")
        if self.eta < 0 or not np.isfinite(self.eta):
            raise ValueError("eta must be finite and nonnegative")
        object.__setattr__(self, "c", tuple(map(tuple, c)))

    @property
    def cm(self):
        return np.asarray(self.c, dtype=float)


def from_cfg(t, pars, name="phonon"):
    """Build a bath from the open_system.bath.pars block of the yaml config.

    Used keys: amp, dip, w0, cs, gam, eta, qd and optionally disp. Without
    disp the dispersion is sqrt(w0^2 + cs^2 q^2) ('gapped'), which contains
    the constant (cs = 0) and the acoustic (w0 = 0) case. gap, ta, tb of the
    block are not used: the vertex takes them from the Band given to cells.
    """
    c = ((pars.get("caa", 1.0), pars.get("cab", 0.0)),
         (pars.get("cba", 0.0), pars.get("cbb", 1.0)))
    return Phonon(t=float(t), amp=pars.get("amp", 1.0), w0=pars.get("w0", 1.0),
                  lam=pars.get("ktf", pars.get("lam", 1.0)),
                  qd=pars.get("qd", np.pi * np.sqrt(2.0)), c=c, name=name,
                  disp=pars.get("disp", "gapped"), cs=pars.get("cs", 0.0),
                  gam=pars.get("gam", pars.get("ew", 0.0)), eta=pars.get("eta", 0.0),
                  dip=pars.get("dip", 0.0))


def auto_L(ns, lam, nq=4, pts=2000, lmin=256, lmax=2048):
    """Fine grid size: enough points per cell and a resolved coupling peak."""
    L = max(lmin, np.sqrt(pts * ns), 2 * np.pi / (nq * lam))
    return int(sfft.next_fast_len(int(np.ceil(min(L, lmax)))))


def _phi(L, lam, qd, nq):
    """Cell averaged phi on the periodic q grid in FFT index order."""
    dq = 2 * np.pi / L
    q0 = sfft.fftfreq(L, 1.0 / L) * dq
    u = ((np.arange(nq) + 0.5) / nq - 0.5) * dq
    ph = np.zeros((L, L))
    for ux in u:
        qx2 = (q0 + ux) ** 2
        for uy in u:
            q2 = qx2[:, None] + ((q0 + uy) ** 2)[None, :]
            ph += np.where(q2 <= qd * qd, q2 / (q2 + lam * lam) ** 2, 0.0)
    return ph / nq**2


def _key(sh, lam, qd, L, nq):
    h = hashlib.sha1()
    for a in (sh.se, np.array([lam, qd, L, nq], dtype=float)):
        h.update(np.ascontiguousarray(a).tobytes())
    if getattr(sh, "nt", 1) > 1:
        h.update(repr(("nt", sh.nt)).encode())  # sectors
    return h.hexdigest()[:16]


def pair_kernel(sh, lam, qd, L=None, nq=4, cache=None, batch=8):
    """Symmetric (ns, ns) pair kernel via FFT convolutions on an L x L grid.

    cache: optional folder; kernels are stored as npz keyed by a hash.
    """
    if L is None:
        L = auto_L(sh.ns, lam, nq)
    path = None
    if cache is not None:
        path = Path(cache) / f"pair_{_key(sh, lam, qd, L, nq)}.npz"
        if path.exists():
            return np.load(path)["A"]

    k = -np.pi + 2 * np.pi * (np.arange(L) + 0.5) / L
    lab = sh.label(np.cos(k)[:, None], np.cos(k)[None, :])
    fp = sfft.rfft2(_phi(L, lam, qd, nq), workers=-1)
    lf = lab.ravel()
    A = np.empty((sh.ns, sh.ns))
    for j0 in range(0, sh.ns, batch):
        js = np.arange(j0, min(j0 + batch, sh.ns))
        chi = (lab[None, :, :] == js[:, None, None]).astype(float)
        cv = sfft.irfft2(fp[None] * sfft.rfft2(chi, workers=-1), s=(L, L), workers=-1)
        for i, j in enumerate(js):
            A[:, j] = np.bincount(lf, weights=cv[i].ravel(), minlength=sh.ns)
    A /= float(L) ** 4
    A = 0.5 * (A + A.T)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, A=A, L=L, lam=lam, qd=qd, nq=nq)
    return A


class Kern(NamedTuple):
    A: np.ndarray  # (nw, ns, ns) velocity part, includes the 1 / w_q of |g|^2
    om: np.ndarray  # (nw,) mean phonon frequency of each bin
    dw: float  # bin width, 0 for a single sharp frequency
    D: np.ndarray = None  # (nw, ns, ns) dipole part


def omega(b, q):
    """Phonon dispersion w_q of bath b at momentum magnitudes q."""
    q = np.asarray(q, dtype=float)
    if b.disp == "constant":
        return np.full_like(q, b.w0)
    if b.disp == "acoustic":
        return b.cs * q
    return np.hypot(b.w0, b.cs * q)


QF = 10  # q fields: dipole, then nine velocity terms


def _qf(b, qx, qy, j):
    """Photon q field j of eqs. 125, 128, 129, with 1 / w_q and without amp.

    j = 0 is the dipole factor (sin^2 + r2 cos^2) / w_q, r2 = (w0 / w_q)^2.
    j = 1..9 are the terms of (X_TE^2 + r2 X_TM^2) / w_q with
    h = sin(p + q / 2); they pair with the source weights of _kf.
    """
    q2 = qx * qx + qy * qy
    qm = np.sqrt(q2)
    om = omega(b, qm)
    ok = (q2 <= b.qd**2) & (om > 0)
    io = np.where(ok, 1.0 / np.where(ok, om, 1.0), 0.0)
    r2 = ((0.0 if b.disp == "acoustic" else b.w0) * io) ** 2
    qs = np.where(qm > 0, qm, 1.0)
    cp, sn = np.where(qm > 0, qx / qs, 1.0), np.where(qm > 0, qy / qs, 0.0)
    pxx = (sn * sn + r2 * cp * cp) * io
    pyy = (cp * cp + r2 * sn * sn) * io
    pxy = 2.0 * sn * cp * (r2 - 1.0) * io
    if j == 0:
        f = pxx
    elif j == 1:
        f = 0.5 * (pxx + pyy)
    elif j < 6:
        g, qa = (pxx, qx) if j < 4 else (pyy, qy)
        f = -0.5 * g * np.cos(qa) if j % 2 == 0 else 0.5 * g * np.sin(qa)
    else:
        hx = (np.cos(0.5 * qx), np.sin(0.5 * qx))[(j - 6) // 2]
        hy = (np.cos(0.5 * qy), np.sin(0.5 * qy))[(j - 6) % 2]
        f = pxy * hx * hy
    return f, om, io


def _kf(k):
    """Source momentum weights (QF, L, L) of the q fields on the k grid."""
    c, s, c2, s2 = np.cos(k), np.sin(k), np.cos(2.0 * k), np.sin(2.0 * k)
    o = np.ones(k.size)
    px = (o, o, c2, s2, o, o, s, s, c, c)
    py = (o, o, o, o, c2, s2, s, c, s, c)
    return np.stack([a[:, None] * b[None, :] for a, b in zip(px, py)])


def _phis(b, L, nq, dw, j=0):
    """Cell averaged photon q field j on the q grid, split into frequency bins.

    Returns (phi (nw, L, L), mean frequency per bin, bin width). The bins
    and their mean frequencies are the same for all j.
    """
    dq = 2 * np.pi / L
    q0 = sfft.fftfreq(L, 1.0 / L) * dq
    u = ((np.arange(nq) + 0.5) / nq - 0.5) * dq
    qm = min(b.qd, np.pi * np.sqrt(2.0))
    lo, hi = float(omega(b, 0.0)), float(omega(b, qm))
    nw = max(1, int(np.ceil((hi - lo) / dw)))
    db = (hi - lo) / nw
    pos = np.arange(L * L).reshape(L, L)
    ph = np.zeros(nw * L * L)
    ws, wo = np.zeros(nw), np.zeros(nw)
    for ux in u:
        for uy in u:
            f, om, io = _qf(b, (q0 + ux)[:, None], (q0 + uy)[None, :], j)
            ix = (np.clip(((om - lo) / db).astype(int), 0, nw - 1) if db > 0
                  else np.zeros(om.shape, dtype=int))
            ph += np.bincount((ix * L * L + pos).ravel(), weights=f.ravel(),
                              minlength=nw * L * L)
            ws += np.bincount(ix.ravel(), weights=io.ravel(), minlength=nw)
            wo += np.bincount(ix.ravel(), weights=(io * om).ravel(), minlength=nw)
    keep = ws > 0
    om = wo[keep] / ws[keep]
    return ph.reshape(nw, L, L)[keep] / nq**2, om, db


def bath_kernel(sh, b, dw, L=None, nq=4, cache=None, batch=8):
    """Binned pair kernel Kern of one bath (see module docstring).

    The frequency range of |q| <= qd is split into bins of width about dw.
    A holds the velocity part, D the dipole part of the photon vertex.
    """
    if L is None:
        L = auto_L(sh.ns, b.lam, nq)
    path = None
    if cache is not None:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(sh.se).tobytes())
        h.update(repr(("photon", b.disp, b.w0, b.cs, b.qd, dw, L, nq)).encode())
        if getattr(sh, "nt", 1) > 1:
            h.update(repr(("st", sh.nt)).encode())  # sectors in signed t
        path = Path(cache) / f"bath_{h.hexdigest()[:16]}.npz"
        if path.exists():
            z = np.load(path)
            return Kern(z["A"], z["om"], float(z["dw"]), z["D"])

    k = -np.pi + 2 * np.pi * (np.arange(L) + 0.5) / L
    lab = sh.label(np.cos(k)[:, None], np.cos(k)[None, :])
    lf = lab.ravel()
    kf = _kf(k)
    A = None
    for jq in range(QF):
        phi, om, db = _phis(b, L, nq, dw, jq)
        fp = sfft.rfft2(phi, workers=-1)
        if A is None:
            A = np.zeros((2, om.size, sh.ns, sh.ns))
        for j0 in range(0, sh.ns, batch):
            js = np.arange(j0, min(j0 + batch, sh.ns))
            chi = sfft.rfft2((lab[None, :, :] == js[:, None, None]) * kf[jq], workers=-1)
            for m in range(om.size):
                cv = sfft.irfft2(fp[m][None] * chi, s=(L, L), workers=-1)
                for i, j in enumerate(js):
                    A[min(jq, 1), m, :, j] += np.bincount(lf, weights=cv[i].ravel(),
                                                          minlength=sh.ns)
    A /= float(L) ** 4
    A = 0.5 * (A + A.transpose(0, 1, 3, 2))
    # FFT round off (|A| < 1e-16 max) where no q of the bin connects I and J
    A[A < 1e-13 * A.max()] = 0.0
    kn = Kern(A[1], om, db, A[0])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, A=A[1], om=om, dw=db, D=A[0])
    return kn


def kernels(sh, band, baths, A=None, dw=None, cache=None):
    """One Kern per bath, baths with the same geometry share one object.

    A may be None (computed), one Kern, a list of Kern, or the original
    (ns, ns) pair_kernel array for constant dispersion baths. The default bin
    width dw is the largest possible energy width of a cell.
    """
    baths = tuple(baths)
    if isinstance(A, Kern):
        return (A,) * len(baths)
    if isinstance(A, (tuple, list)):
        if len(A) != len(baths) or not all(isinstance(k, Kern) for k in A):
            raise ValueError("A must hold one Kern per bath")
        return tuple(A)
    if A is not None:
        if any(b.disp != "constant" for b in baths):
            raise ValueError("a plain pair kernel needs constant dispersion baths")
        A = np.asarray(A)
        return tuple(Kern(A[None] / b.w0, np.array([b.w0]), 0.0) for b in baths)
    if dw is None:
        dw = band.slope * float(sh.se[1] - sh.se[0])
    memo, out = {}, []
    for b in baths:
        key = (b.disp, b.w0 if b.disp != "acoustic" else 0.0,
               b.cs if b.disp != "constant" else 0.0, b.lam, b.qd)
        if key not in memo:
            memo[key] = bath_kernel(sh, b, dw, cache=cache)
        out.append(memo[key])
    return tuple(out)
