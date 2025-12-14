from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class State:
    """Quadrotor state."""

    p: Array
    v: Array
    q: Array
    omega: Array
    omega_m: Array

    def as_vector(self) -> Array:
        return np.concatenate(
            [self.p, self.v, self.q, self.omega, self.omega_m], axis=0
        )

    @staticmethod
    def from_vector(x: Array) -> "State":
        x = np.asarray(x).reshape(-1)
        assert x.shape[0] == 3 + 3 + 4 + 3 + 4
        return State(
            p=x[0:3],
            v=x[3:6],
            q=x[6:10],
            omega=x[10:13],
            omega_m=x[13:17],
        )


@dataclass(frozen=True)
class Wrench:
    """Desired total thrust and body torque."""

    thrust: float
    tau: Array  # (3,)

    def as_vector(self) -> Array:
        return np.array(
            [self.thrust, self.tau[0], self.tau[1], self.tau[2]], dtype=float
        )
