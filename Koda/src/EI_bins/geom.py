"""Phonon bath parameters and the static momentum pair kernel.

A[I, J] = sum_{k in I, p in J} w_k w_p phi(k - p), with
phi(q) = q^2 / (q^2 + lam^2)^2 for |q| <= qd (BZ wrapped q).
The full coupling is |g(q)|^2 = (amp / w0) phi(q), so A depends only on
(shells, lam, qd) and is shared by all baths and all (T, d, m).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.fft as sfft


@dataclass(frozen=True)
class Phonon:
    """Einstein phonon bath coupled to the physical orbitals through c."""

    t: float
    amp: float = 1.0
    w0: float = 1.0
    lam: float = 1.0
    qd: float = np.pi * np.sqrt(2.0)
    c: tuple = ((1.0, 0.0), (0.0, 1.0))
    name: str = "phonon"

    def __post_init__(self):
        c = np.asarray(self.c, dtype=float)
        if c.shape != (2, 2) or not np.isfinite(c).all():
            raise ValueError("c must be a finite 2x2 matrix")
        if not np.allclose(c, c.T):
            raise ValueError("c must be symmetric, c_ab = c_ba")
        if self.t < 0 or self.amp < 0:
            raise ValueError("t and amp must be nonnegative")
        if min(self.w0, self.lam, self.qd) <= 0:
            raise ValueError("w0, lam and qd must be positive")
        object.__setattr__(self, "c", tuple(map(tuple, c)))

    @property
    def cm(self):
        return np.asarray(self.c, dtype=float)


def from_cfg(t, pars, name="phonon"):
    """Build a bath from the config_phonons.yaml style parameter block."""
    c = ((pars.get("caa", 1.0), pars.get("cab", 0.0)),
         (pars.get("cba", 0.0), pars.get("cbb", 1.0)))
    return Phonon(t=float(t), amp=pars.get("amp", 1.0), w0=pars.get("w0", 1.0),
                  lam=pars.get("ktf", pars.get("lam", 1.0)),
                  qd=pars.get("qd", np.pi * np.sqrt(2.0)), c=c, name=name)


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
    lab = sh.cell(np.cos(k)[:, None] + np.cos(k)[None, :])
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
