# src/flexible_moral_hazard/params.py
from __future__ import annotations

import numpy as np

from .model import ubar_two_period_from_index

# Each case is a dict with keys matching solve_with_optimagic(...) arguments.

CASES: dict[str, dict] = {
    "baseline_one_period": {
        "pi": np.array([-1.0, 0.0, 1.0]),
        "p0": np.array([0.5, 0.3, 0.2]),
        "beta": 0.0,
        "delta": 0.0,
        "rho": 1.0,
        "theta": 1.0,
        "eps": 0.05,
        "ubar": float(np.log(0.12)),  # one-period outside option
    },
    "one_period_low_outside_option": {
        "pi": np.array([-1.0, 0.0, 1.0]),
        "p0": np.array([0.5, 0.3, 0.2]),
        "beta": 0.0,
        "delta": 0.0,
        "rho": 1.0,
        "theta": 1.0,
        "eps": 0.05,
        "ubar": float(np.log(0.12)),  # one-period outside option
    },   
    "baseline_two_period": {
        "pi": np.array([-1.0, 0.0, 1.0]),
        "p0": np.array([0.5, 0.3, 0.2]),
        "beta": 1.0,
        "delta": 1.0,
        "rho": 1.0,
        "theta": 1.0,
        "eps": 0.05,
        # Outside option convention: ubar^(1) = ln(u_index), ubar^(2)=(1+delta)*ubar^(1)
        "ubar": ubar_two_period_from_index(u_index=0.12, delta=1.0),
    },
    "two_period_low_outside_option": {
        "pi": np.array([-1.0, 0.0, 1.0]),
        "p0": np.array([0.5, 0.3, 0.2]),
        "beta": 1.0,
        "delta": 1.0,
        "rho": 1.0,
        "theta": 1.0,
        "eps": 0.05,
        "ubar": ubar_two_period_from_index(u_index=0.05, delta=1.0),
    },
    "two_period_impatient_principal": {
        "pi": np.array([-1.0, 0.0, 1.0]),
        "p0": np.array([0.5, 0.3, 0.2]),
        "beta": 0.1,
        "delta": 1.0,
        "rho": 1.0,
        "theta": 1.0,
        "eps": 0.05,
        "ubar": ubar_two_period_from_index(u_index=0.12, delta=1.0),
    },
}
