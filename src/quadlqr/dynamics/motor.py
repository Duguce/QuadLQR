from __future__ import annotations

import numpy as np

Array = np.ndarray


class MotorModel:
    """First-order motor speed dynamics: domega/dt = (omega_cmd - omega)/tau"""

    def __init__(self, tau: float):
        self.tau = float(tau)

    def deriv(self, omega: Array, omega_cmd: Array) -> Array:
        omega = np.asarray(omega, dtype=float).reshape(4)
        omega_cmd = np.asarray(omega_cmd, dtype=float).reshape(4)
        return (omega_cmd - omega) / self.tau
