from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from ..config import Limits, LQRConfig, QuadParams
from ..math.quaternion import q_normalize, q_to_R
from ..math.so3 import vee
from ..types import State, Wrench
from .reference import accel_to_q_and_thrust


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def so3_error(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    """SO(3) attitude error vector (Lee et al.-style): e_R = 0.5 * vee(Rd^T R - R^T Rd)"""
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


@dataclass
class HierarchicalLQR:
    quad: QuadParams
    cfg: LQRConfig
    limits: Limits
    K_outer: np.ndarray
    K_inner: np.ndarray

    @staticmethod
    def build(quad: QuadParams, cfg: LQRConfig, limits: Limits) -> "HierarchicalLQR":
        # Outer: double integrator, u = a_cmd
        Ao = np.block(
            [
                [np.zeros((3, 3)), np.eye(3)],
                [np.zeros((3, 3)), np.zeros((3, 3))],
            ]
        )
        Bo = np.block(
            [
                [np.zeros((3, 3))],
                [np.eye(3)],
            ]
        )
        Qo = np.diag([cfg.Qo_pos] * 3 + [cfg.Qo_vel] * 3)
        Ro = np.diag([cfg.Ro_acc] * 3)
        Ko = lqr_gain(Ao, Bo, Qo, Ro)

        # Inner: small-angle SO(3) error model
        # x = [e_R(3), e_w(3)], e_R_dot ≈ e_w, e_w_dot = J^{-1} tau
        J = quad.J
        Ai = np.block(
            [
                [np.zeros((3, 3)), np.eye(3)],
                [np.zeros((3, 3)), np.zeros((3, 3))],
            ]
        )
        Bi = np.block(
            [
                [np.zeros((3, 3))],
                [np.linalg.inv(J)],
            ]
        )
        Qi = np.diag([cfg.Qi_R] * 3 + [cfg.Qi_w] * 3)
        Ri = np.diag([cfg.Ri_tau] * 3)
        Ki = lqr_gain(Ai, Bi, Qi, Ri)

        return HierarchicalLQR(
            quad=quad, cfg=cfg, limits=limits, K_outer=Ko, K_inner=Ki
        )

    def compute(self, st: State, ref: dict) -> Wrench:
        p, v, q, w = st.p, st.v, q_normalize(st.q), st.omega
        p_d = np.asarray(ref["p_d"], dtype=float).reshape(3)
        v_d = np.asarray(ref.get("v_d", np.zeros(3)), dtype=float).reshape(3)
        a_ff = np.asarray(ref.get("a_ff", np.zeros(3)), dtype=float).reshape(3)

        yaw_d = float(ref.get("yaw_d", self.cfg.yaw_des))
        if not self.cfg.yaw_track:
            yaw_d = float(self.cfg.yaw_des)

        # Outer LQR
        ep = p - p_d
        ev = v - v_d
        xo = np.concatenate([ep, ev], axis=0)
        a_cmd = a_ff - self.K_outer @ xo

        # Map to desired attitude + thrust
        q_d, thrust = accel_to_q_and_thrust(a_cmd, yaw_d, self.quad.m, self.quad.g)

        # Inner LQR on SO(3) error
        R = q_to_R(q)
        Rd = q_to_R(q_d)
        e_R = so3_error(R, Rd)
        e_w = w - np.zeros(3)

        xi = np.concatenate([e_R, e_w], axis=0)
        tau = -self.K_inner @ xi

        # Saturations
        thrust = float(np.clip(thrust, self.limits.thrust_min, self.limits.thrust_max))
        tau = np.clip(tau, -self.limits.tau_max, self.limits.tau_max)

        return Wrench(thrust=thrust, tau=tau)
