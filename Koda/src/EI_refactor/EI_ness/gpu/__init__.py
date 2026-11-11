import jax

jax.config.update("jax_enable_x64", True)

from .rates import check_db, dev, mf, rates, targets
from .solvers import Sol, cur, occ, resid, solve, solve_fixed
