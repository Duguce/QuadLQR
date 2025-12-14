from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TrajConfig


@dataclass(frozen=True)
class Ref:
    p_d: np.ndarray
    v_d: np.ndarray
    a_ff: np.ndarray
    yaw_d: float = 0.0


def hover(t: float, cfg: TrajConfig) -> Ref:
    p = np.array([0.0, 0.0, cfg.hover_z], dtype=float)
    return Ref(p_d=p, v_d=np.zeros(3), a_ff=np.zeros(3), yaw_d=0.0)


def line(t: float, cfg: TrajConfig) -> Ref:
    v = cfg.line_v
    p = np.array([v * t, 0.0, cfg.hover_z], dtype=float)
    v_d = np.array([v, 0.0, 0.0], dtype=float)
    return Ref(p_d=p, v_d=v_d, a_ff=np.zeros(3), yaw_d=0.0)


def circle(t: float, cfg: TrajConfig) -> Ref:
    r = cfg.circle_r
    w = cfg.circle_omega
    p = np.array([r * np.cos(w * t), r * np.sin(w * t), cfg.circle_z], dtype=float)
    v_d = np.array([-r * w * np.sin(w * t), r * w * np.cos(w * t), 0.0], dtype=float)
    a_ff = np.array(
        [-r * w * w * np.cos(w * t), -r * w * w * np.sin(w * t), 0.0], dtype=float
    )
    return Ref(p_d=p, v_d=v_d, a_ff=a_ff, yaw_d=0.0)
