import scipy.constants as sc

EV = sc.e
ANGSTROM = 1e-10
HBAR = sc.hbar
KB = sc.k
C = sc.c
ALPHA = sc.alpha
U = sc.u


def dimensionless_scales(t_a_ev, a_0_angstrom, mass_u=20.0, omega_0_dimless=0.1, c_s_si=2e3, d_angstrom=0.5):
    t_a = t_a_ev * EV
    a_0 = a_0_angstrom * ANGSTROM
    mass = mass_u * U
    v_0 = t_a * a_0 / HBAR
    tau_0 = HBAR / t_a
    c_dimless = C / v_0
    c_s_dimless = c_s_si / v_0
    omega_0_si = omega_0_dimless * t_a / HBAR
    L_dimless = sc.pi * c_dimless / omega_0_dimless
    L_si = L_dimless * a_0
    M_dimless = mass * (a_0 ** 2) * t_a / (HBAR ** 2)
    A_photon = 4.0 * ALPHA * omega_0_dimless
    A_phonon = 1.0 / (2.0 * M_dimless)
    d_dimless = (d_angstrom * ANGSTROM) / a_0
    return {
        "v_0": v_0,
        "tau_0": tau_0,
        "c": c_dimless,
        "c_s": c_s_dimless,
        "omega_0": omega_0_dimless,
        "omega_0_si": omega_0_si,
        "L": L_dimless,
        "L_si": L_si,
        "M": M_dimless,
        "A_photon": A_photon,
        "A_phonon": A_phonon,
        "d": d_dimless,
    }

def energy_to_kelvin(val, t_a_ev=0.1):
    return val * (t_a_ev * EV) / KB

def freq_to_kelvin(val, t_a_ev=0.1):
    return val * HBAR * ((t_a_ev * EV) / HBAR) / KB

def to_si(values, t_a_ev=0.1, a_0_angstrom=5.0, mass_u=20.0):
    t_a = t_a_ev * EV
    a_0 = a_0_angstrom * ANGSTROM
    mass = mass_u * U
    v_0 = t_a * a_0 / HBAR
    tau_0 = HBAR / t_a
    M_dimless = mass * (a_0 ** 2) * t_a / (HBAR ** 2)
    factors = {
        "energy": t_a,
        "kelvin": t_a / KB,
        "frequency": t_a / HBAR,
        "rate": t_a / HBAR,
        "time": tau_0,
        "velocity": v_0,
        "length": a_0,
        "mass": mass / M_dimless,
        "coupling_sq": t_a ** 2,
    }
    out = {}
    for k, v in values.items():
        if k in factors:
            out[k] = v * factors[k]
        elif k == "energy_to_kelvin":
            out[k] = v * t_a / KB
        elif k == "freq_to_kelvin":
            out[k] = v * HBAR * (t_a / HBAR) / KB
    return out
