from __future__ import annotations

import numpy as np

Array = np.ndarray


def compute_metrics(npz: dict) -> dict:
    t: Array = npz["t"]
    X: Array = npz["X"]
    U: Array = npz["U"]
    p_ref: Array = npz["p_ref"]

    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    p = X[:, 0:3]
    e = p - p_ref

    rmse = float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
    max_err = float(np.max(np.linalg.norm(e, axis=1)))

    thrust = U[:, 0]
    tau = U[:, 1:4]
    energy = float(np.sum((thrust**2 + np.sum(tau * tau, axis=1)) * dt))

    peak_thrust = float(np.max(np.abs(thrust)))
    peak_tau = np.max(np.abs(tau), axis=0).tolist()

    return {
        "rmse_pos": rmse,
        "max_pos_err": max_err,
        "energy_u": energy,
        "peak_thrust": peak_thrust,
        "peak_tau": peak_tau,
    }
