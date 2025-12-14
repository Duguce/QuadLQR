from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuadParams:
    # micro quad from your doc: mass ~ 37g
    m: float = 0.037
    g: float = 9.81
    J: np.ndarray = np.diag([1.8e-5, 1.8e-5, 3.2e-5])  # kg*m^2 (reasonable micro-scale)


@dataclass
class MotorParams:
    # First-order motor dynamics: domega/dt = (omega_cmd - omega)/tau
    tau: float = 0.02  # 20ms


@dataclass
class RotorParams:
    # thrust = kf * omega^2 ; yaw torque = km * omega^2
    kf: float = 6.0e-8  # N/(rad/s)^2  (reasonable scale; tune if needed)
    km: float = 1.0e-9  # N*m/(rad/s)^2
    arm: float = 0.046  # m (approx half of diagonal 0.092m)
    # rotor spin directions for yaw: [+ - + -] (X config typical)
    # mixer uses km signs internally, so this is implicit.


@dataclass
class Limits:
    omega_max: float = 2500.0  # rad/s
    omega_min: float = 0.0
    thrust_min: float = 0.0
    thrust_max: float = 1.2  # N  (mg ~ 0.363N)
    tau_max: float = 0.03  # N*m


@dataclass
class DisturbanceConfig:
    """Bounded force (world) and torque (body) disturbances."""

    level: int = 1  # 0 none, 1 medium, 2 strong
    seed: int = 7

    # Force disturbance amps (N)
    force_amp_1: np.ndarray = np.array([0.05, 0.05, 0.04])
    force_amp_2: np.ndarray = np.array([0.10, 0.10, 0.08])
    force_freq_hz: np.ndarray = np.array([0.7, 0.9, 0.5])
    force_noise_sigma_1: float = 0.01
    force_noise_sigma_2: float = 0.02

    # Torque disturbance amps (N*m)
    tau_amp_1: np.ndarray = np.array([0.002, 0.002, 0.0015])
    tau_amp_2: np.ndarray = np.array([0.004, 0.004, 0.003])
    tau_freq_hz: np.ndarray = np.array([1.2, 1.0, 0.8])
    tau_noise_sigma_1: float = 0.0004
    tau_noise_sigma_2: float = 0.0008


@dataclass
class LQRConfig:
    # Outer loop: x=[ep(3), ev(3)], u=a_cmd(3)
    Qo_pos: float = 60.0
    Qo_vel: float = 12.0
    Ro_acc: float = 2.0

    # Inner loop (SO(3) small error): x=[e_R(3), e_w(3)], u=tau(3)
    Qi_R: float = 140.0
    Qi_w: float = 12.0
    Ri_tau: float = 0.6

    yaw_track: bool = False
    yaw_des: float = 0.0


@dataclass
class PIDConfig:
    # Outer position PID -> desired acceleration
    kp_pos: np.ndarray = np.array([2.5, 2.5, 4.0])
    ki_pos: np.ndarray = np.array([0.05, 0.05, 0.08])
    kd_pos: np.ndarray = np.array([2.0, 2.0, 2.6])
    integ_limit: float = 2.0

    # Inner attitude PD -> desired torque (uses SO(3) error vector)
    kp_R: np.ndarray = np.array([0.10, 0.10, 0.08])
    kd_w: np.ndarray = np.array([0.02, 0.02, 0.015])


@dataclass
class SimConfig:
    dt: float = 0.01
    t_hover: float = 30.0
    t_line: float = 30.0
    t_circle: float = 40.0


@dataclass
class TrajConfig:
    hover_z: float = 1.0
    line_v: float = 0.3
    circle_r: float = 1.0
    circle_omega: float = 0.4
    circle_z: float = 1.0


@dataclass
class ExperimentConfig:
    quad: QuadParams = QuadParams()
    motor: MotorParams = MotorParams()
    rotor: RotorParams = RotorParams()
    limits: Limits = Limits()
    disturb: DisturbanceConfig = DisturbanceConfig()
    lqr: LQRConfig = LQRConfig()
    pid: PIDConfig = PIDConfig()
    sim: SimConfig = SimConfig()
    traj: TrajConfig = TrajConfig()
