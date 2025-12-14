from __future__ import annotations

import numpy as np

Array = np.ndarray


def hat(w: Array) -> Array:
    wx, wy, wz = w
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=float)


def vee(W: Array) -> Array:
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


def clamp_norm(v: Array, max_norm: float) -> Array:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)
