from __future__ import annotations

import os

import numpy as np

from ..config import ExperimentConfig
from ..control import BaselinePID, HierarchicalLQR, Mixer
from ..dynamics import MotorModel, QuadrotorPlant
from ..types import State
from .integrator import rk4_step

Array = np.ndarray


def run_case(
    cfg: ExperimentConfig,
    name: str,
    ref_fn,
    controller: str,
    t_final: float,
    outdir: str,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    plant = QuadrotorPlant(cfg.quad, cfg.rotor, cfg.disturb)
    plant.reset_rng(cfg.disturb.seed)

    motor = MotorModel(cfg.motor.tau)
    mixer = Mixer(
        cfg.rotor.kf,
        cfg.rotor.km,
        cfg.rotor.arm,
        cfg.limits.omega_min,
        cfg.limits.omega_max,
    )

    lqr = HierarchicalLQR.build(cfg.quad, cfg.lqr, cfg.limits)
    pid = BaselinePID.build(cfg.quad, cfg.pid, cfg.limits)
    pid.reset()

    dt = cfg.sim.dt
    n = int(np.floor(t_final / dt)) + 1
    t = np.linspace(0.0, t_final, n)

    # Initial state
    x0 = State(
        p=np.array([0.2, -0.2, cfg.traj.hover_z - 0.1], dtype=float),
        v=np.zeros(3, dtype=float),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        omega=np.zeros(3, dtype=float),
        omega_m=np.ones(4, dtype=float) * 1200.0,
    ).as_vector()

    X = np.zeros((n, x0.shape[0]), dtype=float)
    U = np.zeros((n, 4), dtype=float)  # desired wrench [T, tau_x, tau_y, tau_z]
    OM = np.zeros((n, 4), dtype=float)  # motor speeds
    OM_CMD = np.zeros((n, 4), dtype=float)
    P_REF = np.zeros((n, 3), dtype=float)
    V_REF = np.zeros((n, 3), dtype=float)

    x = x0.copy()

    def closed_loop_rhs(tk: float, xk: Array) -> Array:
        """Return derivative of full state. We integrate rigid body with plant.f, plus motor dynamics."""
        st = State.from_vector(xk)

        ref = ref_fn(tk, cfg.traj)
        ref_dict = {
            "p_d": ref.p_d,
            "v_d": ref.v_d,
            "a_ff": ref.a_ff,
            "yaw_d": ref.yaw_d,
        }

        if controller.lower() == "lqr":
            wrench = lqr.compute(st, ref_dict)
        elif controller.lower() == "pid":
            wrench = pid.compute(st, ref_dict, dt)
        else:
            raise ValueError(f"Unknown controller: {controller}")

        # allocation: wrench -> omega_cmd
        omega_cmd = mixer.allocate(wrench.thrust, wrench.tau)

        # motor derivative
        domega = motor.deriv(st.omega_m, omega_cmd)

        # rigid-body derivative from plant: uses omega_m in state
        xdot = plant.f(tk, xk)
        xdot = xdot.copy()
        xdot[13:17] = domega  # overwrite motor dot (plant uses zeros placeholder)

        return xdot

    for k in range(n):
        tk = float(t[k])
        st = State.from_vector(x)

        ref = ref_fn(tk, cfg.traj)
        ref_dict = {
            "p_d": ref.p_d,
            "v_d": ref.v_d,
            "a_ff": ref.a_ff,
            "yaw_d": ref.yaw_d,
        }

        # compute controller output for logging
        if controller.lower() == "lqr":
            wrench = lqr.compute(st, ref_dict)
        else:
            wrench = pid.compute(st, ref_dict, dt)
        omega_cmd = mixer.allocate(wrench.thrust, wrench.tau)

        X[k, :] = x
        U[k, :] = wrench.as_vector()
        OM[k, :] = st.omega_m
        OM_CMD[k, :] = omega_cmd
        P_REF[k, :] = ref.p_d
        V_REF[k, :] = ref.v_d

        if k < n - 1:
            x = rk4_step(closed_loop_rhs, tk, x, dt)
            x = plant.post_process(x)

    path = os.path.join(logdir, f"{name}__{controller}.npz")
    np.savez_compressed(
        path,
        t=t,
        X=X,
        U=U,
        omega=OM,
        omega_cmd=OM_CMD,
        p_ref=P_REF,
        v_ref=V_REF,
    )
    return path
