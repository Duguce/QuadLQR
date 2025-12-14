from __future__ import annotations

import numpy as np

from ..config import DisturbanceConfig, QuadParams, RotorParams
from ..math.quaternion import omega_to_qdot, q_normalize, q_to_R
from ..types import State

Array = np.ndarray


class QuadrotorPlant:
    """Nonlinear 6DOF rigid-body + quaternion attitude."""

    def __init__(
        self, quad: QuadParams, rotor: RotorParams, disturb: DisturbanceConfig
    ):
        self.quad = quad
        self.rotor = rotor
        self.disturb = disturb
        self.rng = np.random.default_rng(disturb.seed)

    def reset_rng(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self.disturb.seed
        self.rng = np.random.default_rng(seed)

    def _disturbance(self, t: float) -> tuple[Array, Array]:
        """Return (force_world, torque_body)."""
        cfg = self.disturb
        if cfg.level <= 0:
            return np.zeros(3), np.zeros(3)

        if cfg.level == 1:
            f_amp = cfg.force_amp_1
            f_sig = cfg.force_noise_sigma_1
            tau_amp = cfg.tau_amp_1
            tau_sig = cfg.tau_noise_sigma_1
        else:
            f_amp = cfg.force_amp_2
            f_sig = cfg.force_noise_sigma_2
            tau_amp = cfg.tau_amp_2
            tau_sig = cfg.tau_noise_sigma_2

        f = f_amp * np.sin(
            2.0 * np.pi * cfg.force_freq_hz * t + np.array([0.0, 0.7, 1.1])
        )
        f += f_sig * self.rng.standard_normal(3)

        tau = tau_amp * np.sin(
            2.0 * np.pi * cfg.tau_freq_hz * t + np.array([0.3, 1.0, 0.2])
        )
        tau += tau_sig * self.rng.standard_normal(3)
        return f, tau

    def wrench_from_omega(self, omega_m: Array) -> tuple[float, Array]:
        """Compute (thrust, tau_body) from rotor speeds omega_m."""
        omega_m = np.asarray(omega_m, dtype=float).reshape(4)
        kf, km, arm = self.rotor.kf, self.rotor.km, self.rotor.arm
        w2 = omega_m**2

        # X configuration mixing consistent with typical indexing:
        # [w1,w2,w3,w4] with yaw torque signs [+,-,+,-]
        T = kf * np.sum(w2)

        tau_x = arm * kf * (w2[1] - w2[3])
        tau_y = arm * kf * (-w2[0] + w2[2])
        tau_z = km * (w2[0] - w2[1] + w2[2] - w2[3])

        return float(T), np.array([tau_x, tau_y, tau_z], dtype=float)

    def f(self, t: float, x: Array) -> Array:
        """Full state derivative for x=[p(3), v(3), q(4), omega(3), omega_m(4)]."""
        st = State.from_vector(x)
        m, g, J = self.quad.m, self.quad.g, self.quad.J

        q = q_normalize(st.q)
        R = q_to_R(q)

        T, tau = self.wrench_from_omega(st.omega_m)
        f_w, tau_d = self._disturbance(t)

        # Translational dynamics (world)
        p_dot = st.v
        v_dot = (
            (R @ np.array([0.0, 0.0, T], dtype=float)) / m
            - np.array([0.0, 0.0, g], dtype=float)
            + f_w / m
        )

        # Rotational dynamics (body)
        q_dot = omega_to_qdot(q, st.omega)
        omega = st.omega
        tau_total = tau + tau_d
        omega_dot = np.linalg.solve(J, tau_total - np.cross(omega, J @ omega))

        # omega_m derivative handled by motor model externally; placeholder zeros here
        omega_m_dot = np.zeros(4, dtype=float)

        xdot = np.concatenate([p_dot, v_dot, q_dot, omega_dot, omega_m_dot], axis=0)
        return xdot

    @staticmethod
    def post_process(x: Array) -> Array:
        """Normalize quaternion after integration."""
        st = State.from_vector(x)
        q = q_normalize(st.q)
        return State(
            p=st.p, v=st.v, q=q, omega=st.omega, omega_m=st.omega_m
        ).as_vector()
