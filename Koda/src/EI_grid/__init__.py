"""Bath driven excitonic insulator on a fixed Cartesian k grid.

Rate matrices of eqs. 186, 187 with the photon vertex of eq. 125, and a
Newton solver for the steady state in the multipliers mu of eq. 193.
"""

from .backend import NAME as backend
from .bath import Bath, bose, klor, lor, nlor, om_q, pack, xbose
from .vertex import Geom, coh, make_geom, pol, vert
from .rates import (Prob, State, cur, dense_rates, dissipator, make_prob, mf,
                    parts, rate_mat, rate_one, tgt)
from .solve import Out, jac, newton, pair_chi, residual, solve_ness

__all__ = [
    "backend", "Bath", "bose", "klor", "lor", "nlor", "om_q", "pack", "xbose",
    "Geom", "coh", "make_geom", "pol", "vert",
    "Prob", "State", "cur", "dense_rates", "dissipator", "make_prob", "mf", "parts",
    "rate_mat", "rate_one", "tgt",
    "Out", "jac", "newton", "pair_chi", "residual", "solve_ness",
]
