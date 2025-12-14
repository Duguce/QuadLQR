from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Limits, PIDConfig, QuadParams
from ..math.quaternion import q_normalize, q_to_R
from ..types import State, Wrench
from .lqr import so3_error
from .reference import accel_to_q_and_thrust


@dataclass
class BaselinePID:
    quad: QuadParams
    cfg: PIDConfig
    limits: Limits
    integ_ep: np.ndarray

    @staticmethod
    def build(quad: QuadParams, cfg: PIDConfig, limits: Limits) -> "BaselinePID":
        return BaselinePID(
            quad=quad, cfg=cfg, limits=limits, integ_ep=np.zeros(3, dtype=float)
        )

    def reset(self) -> None:
        self.integ_ep[:] = 0.0

    def compute(self, st: State, ref: dict, dt: float) -> Wrench:
        p, v, q, w = st.p, st.v, q_normalize(st.q), st.omega
        p_d = np.asarray(ref["p_d"], dtype=float).reshape(3)
        v_d = np.asarray(ref.get("v_d", np.zeros(3)), dtype=float).reshape(3)
        a_ff = np.asarray(ref.get("a_ff", np.zeros(3)), dtype=float).reshape(3)
        yaw_d = float(ref.get("yaw_d", 0.0))

        ep = p - p_d
        ev = v - v_d

        self.integ_ep += ep * dt
        self.integ_ep = np.clip(
            self.integ_ep, -self.cfg.integ_limit, self.cfg.integ_limit
        )

        a_cmd = (
            a_ff
            - self.cfg.kp_pos * ep
            - self.cfg.kd_pos * ev
            - self.cfg.ki_pos * self.integ_ep
        )

        q_d, thrust = accel_to_q_and_thrust(a_cmd, yaw_d, self.quad.m, self.quad.g)

        R = q_to_R(q)
        Rd = q_to_R(q_d)
        e_R = so3_error(R, Rd)
        e_w = w - np.zeros(3)

        tau = -self.cfg.kp_R * e_R - self.cfg.kd_w * e_w

        thrust = float(np.clip(thrust, self.limits.thrust_min, self.limits.thrust_max))
        tau = np.clip(tau, -self.limits.tau_max, self.limits.tau_max)

        return Wrench(thrust=thrust, tau=tau)
