from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from quadlqr.config import ExperimentConfig
from quadlqr.metrics import compute_metrics
from quadlqr.plotting import (
    plot_hover_errors,
    plot_inputs,
    plot_motor_speeds,
    plot_traj_xy,
)
from quadlqr.sim.runner import run_case
from quadlqr.sim.scenarios import circle, hover, line


def main() -> None:
    cfg = ExperimentConfig()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("outputs", ts)
    figdir = os.path.join(outdir, "figs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    rows = []

    # Exp-1 Hover (LQR)
    p1 = run_case(cfg, "Exp1_Hover", hover, "lqr", cfg.sim.t_hover, outdir)
    npz = np.load(p1)
    rows.append({"exp": "Exp1_Hover", "controller": "LQR", **compute_metrics(npz)})
    plot_hover_errors(npz, figdir, "Exp1_Hover__LQR")
    plot_inputs(npz, figdir, "Exp1_Hover__LQR")
    plot_motor_speeds(npz, figdir, "Exp1_Hover__LQR")

    # Exp-2 Line (LQR)
    p2 = run_case(cfg, "Exp2_Line", line, "lqr", cfg.sim.t_line, outdir)
    npz = np.load(p2)
    rows.append({"exp": "Exp2_Line", "controller": "LQR", **compute_metrics(npz)})
    plot_traj_xy(npz, figdir, "Exp2_Line__LQR", "Line Tracking (XY)")
    plot_inputs(npz, figdir, "Exp2_Line__LQR")
    plot_motor_speeds(npz, figdir, "Exp2_Line__LQR")

    # Exp-3 Circle (LQR)
    p3 = run_case(cfg, "Exp3_Circle", circle, "lqr", cfg.sim.t_circle, outdir)
    npz = np.load(p3)
    rows.append({"exp": "Exp3_Circle", "controller": "LQR", **compute_metrics(npz)})
    plot_traj_xy(npz, figdir, "Exp3_Circle__LQR", "Circle Tracking (XY)")
    plot_inputs(npz, figdir, "Exp3_Circle__LQR")
    plot_motor_speeds(npz, figdir, "Exp3_Circle__LQR")

    # Exp-4 Circle Compare (PID baseline)
    p4 = run_case(cfg, "Exp4_CircleCompare", circle, "pid", cfg.sim.t_circle, outdir)
    npz = np.load(p4)
    rows.append(
        {"exp": "Exp4_CircleCompare", "controller": "PID", **compute_metrics(npz)}
    )
    plot_traj_xy(npz, figdir, "Exp4_Circle__PID", "Circle Tracking (XY) - PID")
    plot_inputs(npz, figdir, "Exp4_Circle__PID")
    plot_motor_speeds(npz, figdir, "Exp4_Circle__PID")

    # Save metrics
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[OK] Done. Outputs: {outdir}")


if __name__ == "__main__":
    main()
