from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray


def rk4_step(
    f: Callable[[float, Array], Array], t: float, x: Array, dt: float
) -> Array:
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
