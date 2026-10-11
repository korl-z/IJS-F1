"""Bosonic bath and the scalar factors of the rates, eqs. 180, 181, 185, 186.

x is always the electronic energy difference Omega = E_final - E_initial,
om is the boson frequency om_q of the transferred momentum q = k - k'.
"""

from dataclasses import dataclass

import numpy as np

from .backend import xp


@dataclass(frozen=True)
class Bath:
    """One bath: temperature, dispersion sqrt(w0^2 + c^2 q^2), line width, coupling."""

    t: float
    w0: float
    c: float
    gam: float
    amp: float = 1.0
    name: str = "bath"

    def __post_init__(self):
        z = np.asarray((self.t, self.w0, self.c, self.gam, self.amp), dtype=float)
        if not np.isfinite(z).all():
            raise ValueError("bath parameters must be finite")
        if self.t <= 0 or self.gam <= 0:
            raise ValueError("t and gam must be positive")
        if min(self.w0, self.c, self.amp) < 0:
            raise ValueError("w0, c and amp must be nonnegative")


def pack(bs, t=None):
    """Bath parameters as rows (t, w0, c, gam, amp); t overrides the temperatures."""
    bs = tuple(bs)
    if not bs:
        raise ValueError("at least one bath is required")
    tt = [b.t for b in bs] if t is None else np.broadcast_to(t, (len(bs),))
    z = [(float(x), b.w0, b.c, b.gam, b.amp) for x, b in zip(tt, bs)]
    return xp.asarray(z, dtype=xp.float64)


def om_q(q, w0, c):
    """Boson dispersion for momentum magnitude q."""
    return xp.sqrt(w0 * w0 + xp.square(c * q))


def bose(x, t):
    """Bose factor N(x) for either sign, N(-x) = -1 - N(x), eq. 185; inf at x = 0."""
    ax = xp.abs(x) / t
    den = -xp.expm1(-ax)
    den = xp.where(den > 0.0, den, 1.0)
    n = xp.where(x > 0.0, xp.exp(-ax), -1.0) / den
    return xp.where(x != 0.0, n, xp.inf)


def xbose(x, t):
    """x N(x), regular everywhere; equals t at x = 0."""
    ax = xp.abs(x) / t
    sm = ax < 1.0e-6
    ay = xp.where(sm, 1.0, ax)
    y = xp.where(x > 0.0, x * xp.exp(-ay), -x) / (-xp.expm1(-ay))
    return xp.where(sm, t - 0.5 * x, y)


def lor(x, om, gam):
    """Lorentzian A_q(x) = 2 gam / ((x - om)^2 + gam^2), eq. 180; A_q(-x) is eq. 181."""
    return 2.0 * gam / (xp.square(x - om) + gam * gam)


def klor(x, om, gam, t):
    """N(x) [A_q(x) - A_q(-x)] / om, the bracket of eq. 186 over one common
    denominator: 8 gam x N(x) / (((x - om)^2 + gam^2) ((x + om)^2 + gam^2))."""
    dm = xp.square(x - om) + gam * gam
    dp = xp.square(x + om) + gam * gam
    return 8.0 * gam * xbose(x, t) / (dm * dp)


def nlor(x, om, gam, t):
    """N(x) [A_q(x) - A_q(-x)], the full scalar factor of eq. 186."""
    return om * klor(x, om, gam, t)
