"""First order photon vertex on the k grid, eqs. 125, 128, 129, 157.

Pair arrays have rows k (final) and columns k' (initial), q = k - k' wrapped
to the first BZ. States flatten as (alpha k, beta k), alpha is the upper band.

With the rotation of eq. 32 the vertex element of one polarization is
  M~ = g_q (X C1 + i P C2),  g_q^2 = amp / om_q,
  X = eps . (sin(k'_x + q_x/2), sin(k'_y + q_y/2)),  P = eps . d,
where C1 and C2 depend only on u, v and the bare bands. The real part comes
from the velocity terms, the imaginary part from the dipole terms, so
|M~|^2 = g_q^2 (X^2 C1^2 + P^2 C2^2) and only X^2, P^2 are stored.
"""

from typing import Any, NamedTuple

import numpy as np

from .backend import xp
from .bath import om_q

TINY = 1.0e-300


class Geom(NamedTuple):
    q: Any  # wrapped k - k', (N, N, 2)
    qm: Any  # |q|, (N, N)
    x2: Any  # X^2 for unit TE and TM polarization, (2, N, N)
    p2: Any  # P^2 for unit TE and TM polarization, (2, N, N)
    ta: float
    tb: float
    dip: float
    th: float


def make_geom(bd, ta, tb, dip, th=0.0, a0=1.0):
    """Pair geometry for bands bd, dipole d = dip (cos th, sin th, 0).

    TE is eq. 128; TM is eq. 129 without om_0/om_q, which depends on the bath
    and is added in vert. Pairs with a q component on the zone edge are
    averaged over the two equivalent images, which keeps |M~|^2 symmetric.
    At q = 0 the angle is phi = 0.
    """
    k = np.asarray(bd.k, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2:
        raise ValueError("bd.k must have shape (N, 2)")
    per = 2.0 * np.pi / a0
    q = k[:, None, :] - k[None, :, :]
    q = (q + 0.5 * per) % per - 0.5 * per
    ed = np.abs(np.abs(q) - 0.5 * per) < 1.0e-9 * per
    sg = ((1, 1), (-1, 1), (1, -1), (-1, -1)) if ed.any() else ((1, 1),)
    x2 = np.zeros((2,) + q.shape[:2])
    p2 = np.zeros_like(x2)
    for s in sg:
        qi = np.where(ed, 0.5 * per * np.asarray(s, dtype=float), q)
        phi = np.arctan2(qi[..., 1], qi[..., 0])
        cp, sp = np.cos(phi), np.sin(phi)
        # velocity direction at the midpoint k' + q/2
        hx = a0 * np.sin(a0 * (k[None, :, 0] + 0.5 * qi[..., 0]))
        hy = a0 * np.sin(a0 * (k[None, :, 1] + 0.5 * qi[..., 1]))
        x2[0] += (-sp * hx + cp * hy) ** 2
        x2[1] += (cp * hx + sp * hy) ** 2
        p2[0] += (dip * np.sin(th - phi)) ** 2
        p2[1] += (dip * np.cos(th - phi)) ** 2
    x2 /= len(sg)
    p2 /= len(sg)
    q = np.where(ed, -0.5 * per, q)
    qm = np.hypot(q[..., 0], q[..., 1])
    return Geom(q, qm, x2, p2, float(ta), float(tb), float(dip), float(th))


def coh(st, pk):
    """Coherence amplitudes C1 (velocity) and C2 (dipole) of eq. 157.

    Each is a tuple of (N, N) arrays in the order (aa, ab, ba, bb) of the
    band pair (final, initial); a = alpha, b = beta.
    """
    u, v = st.u, st.v
    uu = u[:, None] * u[None, :]
    uv = u[:, None] * v[None, :]
    vu = v[:, None] * u[None, :]
    vv = v[:, None] * v[None, :]
    # bare band differences of the dipole terms of eq. 125
    eab = pk.ea[:, None] - pk.eb[None, :]
    eba = pk.eb[:, None] - pk.ea[None, :]
    a, b = 2.0 * pk.ta, 2.0 * pk.tb
    c1 = (a * uu + b * vv, a * uv - b * vu, a * vu - b * uv, a * vv + b * uu)
    c2 = (uv * eab + vu * eba, uu * eab - vv * eba,
          uu * eba - vv * eab, vu * eab + uv * eba)
    return c1, c2


def pol(pk, w0, c):
    """Boson frequency and polarization summed X^2, P^2 for one dispersion."""
    om = om_q(pk.qm, w0, c)
    # TM carries om_0 / om_q, eq. 129
    r2 = xp.square(w0 / xp.maximum(om, TINY))
    return om, pk.x2[0] + r2 * pk.x2[1], pk.p2[0] + r2 * pk.p2[1]


def vert(st, pk, w0, c, amp):
    """om_q sum_pol |M~|^2 as a (2N, 2N) matrix, and om_q tiled to that shape.

    The factor 1 / om_q of g_q^2 is left out here; it cancels against the om_q
    of the Lorentzian difference in the rate (see bath.klor).
    """
    om, xx, pp = pol(pk, w0, c)
    c1, c2 = coh(st, pk)
    s = [amp * (xx * c1[i] ** 2 + pp * c2[i] ** 2) for i in range(4)]
    return xp.block([[s[0], s[1]], [s[2], s[3]]]), xp.tile(om, (2, 2))
