from __future__ import annotations

import os

import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_hover_errors(npz: dict, outdir: str, tag: str) -> str:
    _ensure_dir(outdir)
    t = npz["t"]
    X = npz["X"]
    p_ref = npz["p_ref"]
    e = X[:, 0:3] - p_ref

    fig = plt.figure()
    plt.plot(t, e[:, 0], label="e_x")
    plt.plot(t, e[:, 1], label="e_y")
    plt.plot(t, e[:, 2], label="e_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (m)")
    plt.legend()
    fig.tight_layout()
    path = os.path.join(outdir, f"Fig_HoverError__{tag}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_traj_xy(npz: dict, outdir: str, tag: str, title: str) -> str:
    _ensure_dir(outdir)
    X = npz["X"]
    p_ref = npz["p_ref"]

    fig = plt.figure()
    plt.plot(p_ref[:, 0], p_ref[:, 1], label="ref")
    plt.plot(X[:, 0], X[:, 1], label="actual")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    fig.tight_layout()
    path = os.path.join(outdir, f"Fig_TrajXY__{tag}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_inputs(npz: dict, outdir: str, tag: str) -> str:
    _ensure_dir(outdir)
    t = npz["t"]
    U = npz["U"]

    fig = plt.figure()
    plt.plot(t, U[:, 0], label="T")
    plt.plot(t, U[:, 1], label="tau_x")
    plt.plot(t, U[:, 2], label="tau_y")
    plt.plot(t, U[:, 3], label="tau_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired wrench")
    plt.legend()
    fig.tight_layout()
    path = os.path.join(outdir, f"Fig_Inputs__{tag}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_motor_speeds(npz: dict, outdir: str, tag: str) -> str:
    _ensure_dir(outdir)
    t = npz["t"]
    om = npz["omega"]
    omc = npz["omega_cmd"]

    fig = plt.figure()
    for i in range(4):
        plt.plot(t, om[:, i], label=f"omega{i + 1}")
    for i in range(4):
        plt.plot(t, omc[:, i], linestyle="--", label=f"omega{i + 1}_cmd")
    plt.xlabel("Time (s)")
    plt.ylabel("Motor speed (rad/s)")
    plt.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"Fig_MotorSpeeds__{tag}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_traj_xy_compare(
    npz_lqr: dict,
    npz_pid: dict,
    outdir: str,
    tag: str,
    title: str = "Circle Tracking Comparison (XY)",
) -> str:
    """Plot reference, LQR actual, and PID actual trajectories on the same XY figure."""
    _ensure_dir(outdir)

    # Reference (should be identical in both npz)
    p_ref = npz_lqr["p_ref"]

    # Actual trajectories
    X_lqr = npz_lqr["X"]
    X_pid = npz_pid["X"]

    fig = plt.figure()
    plt.plot(
        p_ref[:, 0],
        p_ref[:, 1],
        "k--",
        linewidth=2.0,
        label="Reference",
    )
    plt.plot(
        X_lqr[:, 0],
        X_lqr[:, 1],
        linewidth=2.0,
        label="LQR",
    )
    plt.plot(
        X_pid[:, 0],
        X_pid[:, 1],
        linewidth=2.0,
        label="PID",
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    fig.tight_layout()

    path = os.path.join(outdir, f"Fig_TrajXY_Compare__{tag}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path
