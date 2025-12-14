from __future__ import annotations

import numpy as np

Array = np.ndarray


class Mixer:
    """Allocate desired [T, tau] to rotor speed commands via omega^2 mixing."""

    def __init__(
        self, kf: float, km: float, arm: float, omega_min: float, omega_max: float
    ):
        self.kf = float(kf)
        self.km = float(km)
        self.arm = float(arm)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

        kf, km, arm = self.kf, self.km, self.arm
        self.M = np.array(
            [
                [kf, kf, kf, kf],
                [0.0, arm * kf, 0.0, -arm * kf],
                [-arm * kf, 0.0, arm * kf, 0.0],
                [km, -km, km, -km],
            ],
            dtype=float,
        )
        self.M_inv = np.linalg.inv(self.M)

    def allocate(self, thrust: float, tau: Array) -> Array:
        u = np.array(
            [float(thrust), float(tau[0]), float(tau[1]), float(tau[2])], dtype=float
        )
        w2 = self.M_inv @ u
        # enforce non-negative
        w2 = np.clip(w2, 0.0, None)
        w = np.sqrt(w2)
        w = np.clip(w, self.omega_min, self.omega_max)
        return w
