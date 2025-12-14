from __future__ import annotations

import numpy as np

from ..math.quaternion import R_to_q

Array = np.ndarray


def accel_to_R_des(a_cmd: Array, yaw_des: float, g: float) -> Array:
    """Geometric mapping from desired acceleration (world) to desired rotation matrix."""
    a_cmd = np.asarray(a_cmd, dtype=float).reshape(3)
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    a_total = a_cmd + g * e3
    norm = float(np.linalg.norm(a_total))
    if norm < 1e-6:
        norm = 1e-6
    b3 = a_total / norm

    cpsi, spsi = np.cos(yaw_des), np.sin(yaw_des)
    b1_des_world = np.array([cpsi, spsi, 0.0], dtype=float)

    # make b2 = b3 x b1_des_world, then b1 = b2 x b3
    b2 = np.cross(b3, b1_des_world)
    n2 = float(np.linalg.norm(b2))
    if n2 < 1e-6:
        # degenerate case: choose arbitrary orthogonal
        b2 = np.cross(b3, np.array([0.0, 1.0, 0.0], dtype=float))
        n2 = float(np.linalg.norm(b2))
        if n2 < 1e-6:
            b2 = np.cross(b3, np.array([1.0, 0.0, 0.0], dtype=float))
            n2 = float(np.linalg.norm(b2))
    b2 = b2 / n2
    b1 = np.cross(b2, b3)

    R = np.column_stack([b1, b2, b3])
    return R


def accel_to_q_and_thrust(
    a_cmd: Array, yaw_des: float, m: float, g: float
) -> tuple[Array, float]:
    """Return desired quaternion q_d (body->world) and desired thrust magnitude."""
    R = accel_to_R_des(a_cmd, yaw_des, g)
    q_d = R_to_q(R)

    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    a_total = np.asarray(a_cmd, dtype=float).reshape(3) + g * e3
    thrust = m * float(np.linalg.norm(a_total))
    return q_d, thrust
