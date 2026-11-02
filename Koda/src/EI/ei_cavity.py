from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

F = jnp.float64


@dataclass(frozen=True)
class CavityBath:
    """Planar cavity photons, q integrated analytically (vertical transitions).

    Only beta_k <-> alpha_k at the same k, in plane TE1 + TM1 modes at L/2.
    """

    t: float  # photon temperature
    w0: float  # cavity cutoff omega_0
    gam: float  # line width
    ta: float
    tb: float
    c: float = 3945.0  # light speed in units t_a a0 / hbar
    amp: float | None = None  # None: 4 alpha w0
    alpha: float = 1.0 / 137.036
    a0: float = 1.0
    name: str = "cavity"

    def __post_init__(self):
        if self.amp is None:
            object.__setattr__(self, "amp", 4.0 * self.alpha * self.w0)
        z = np.asarray((self.t, self.w0, self.gam, self.ta, self.tb,
                        self.c, self.amp, self.a0))
        if not np.isfinite(z).all():
            raise ValueError("cavity parameters must be finite")
        if min(self.w0, self.gam, self.c, self.a0) <= 0.0:
            raise ValueError("w0, gam, c and a0 must be positive")
        if min(self.t, self.amp) < 0.0:
            raise ValueError("t and amp must be nonnegative")


def stack(cb):
    """Rows (t, w0, gam, amp, c, ta, tb, a0) of the cavity baths, or None."""
    if not cb:
        return None
    return jnp.asarray([(b.t, b.w0, b.gam, b.amp, b.c, b.ta, b.tb, b.a0)
                        for b in cb], dtype=F)


def kern(om, w0, gam):
    """Bracket of J: 4 gam Om int_w0^inf (w + w0^2 / w) / (d1 d2) dw."""
    z4 = (om**2 + gam**2) ** 2
    a = jnp.arctan((om - w0) / gam) + jnp.arctan((om + w0) / gam)
    l = jnp.log(((om - w0) ** 2 + gam**2) * ((om + w0) ** 2 + gam**2) / w0**4)
    return (1.0 + w0**2 * (om**2 - gam**2) / z4) * a + gam * om * w0**2 / z4 * l


def _bose(w, t):
    x = w / jnp.where(t > 0.0, t, 1.0)
    n = 1.0 / jnp.maximum(jnp.expm1(x), F(1.0e-300))
    return jnp.where((t > 0.0) & (w > 0.0), n, 0.0)


def local(e, u, v, k, cav):
    """Rates up (beta_k -> alpha_k) and dn (alpha_k -> beta_k), shape (n,).

    J = amp / (2 pi c^2) a0^2 (sin^2 kx + sin^2 ky) [2 (ta - tb) u v]^2 K(Om),
    Om = E_alpha - E_beta, up = J N(Om), dn = J (1 + N(Om)).
    """
    om = (e[0] - e[1])[None, :]
    t, w0, gam, amp, c, ta, tb, a0 = (cav[:, i, None] for i in range(8))
    hv = a0**2 * (jnp.sin(a0 * k[None, :, 0]) ** 2 + jnp.sin(a0 * k[None, :, 1]) ** 2)
    cc = jnp.square(2.0 * (ta - tb) * (u * v)[None, :])
    j = amp / (2.0 * jnp.pi * c**2) * hv * cc * kern(om, w0, gam)
    nb = _bose(om, t)
    return jnp.sum(j * nb, axis=0), jnp.sum(j * (1.0 + nb), axis=0)


def loc_cols(e, u, v, k, w, cav, j):
    """Local photon block as columns j of R[dest, src] for weights w.

    The solvers multiply R by the source weight, so the entry is rate / w.
    """
    n = w.size
    up, dn = local(e, u, v, k, cav)
    rin = jnp.concatenate((up, dn)) / jnp.tile(w, 2)  # into alpha_k, into beta_k
    rows = (j + n) % (2 * n)  # partner state of each source column
    z = jnp.zeros((2 * n, j.size), dtype=rin.dtype)
    return z.at[rows, jnp.arange(j.size)].set(rin[rows])


def rate_2x2(st, cb, k):
    """Per k 2x2 matrices [dest, src] in (alpha, beta), for inspection."""
    up, dn = local(st.e, st.u, st.v, jnp.asarray(k, dtype=F), stack(tuple(cb)))
    z = jnp.zeros_like(up)
    return jnp.stack((jnp.stack((z, up), -1), jnp.stack((dn, z), -1)), -2)
