"""
Shared action preprocessing utilities for training datasets.

This module is intentionally small and fail-hard:
- no silent fallbacks for missing timestamps or bad shapes
- strict validation of action conventions
- explicit interpolation mode handling
"""

from typing import Tuple

import numpy as np


def _require_2d(x: np.ndarray, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError(f"{name} must have positive shape, got {x.shape}")


def _require_1d(x: np.ndarray, name: str) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    if x.shape[0] <= 0:
        raise ValueError(f"{name} must have positive length, got {x.shape[0]}")


def _require_monotonic_increasing(ts_s: np.ndarray, name: str) -> None:
    d = np.diff(ts_s)
    if np.any(d <= 0.0):
        raise ValueError(f"{name} must be strictly increasing, got min diff {float(np.min(d)):.8f}")


def _check_interp_mode(mode: str) -> None:
    if mode not in ["linear", "nearest", "cubic"]:
        raise ValueError(f"interp_mode must be linear/nearest/cubic, got '{mode}'")


def rot6d_to_matrix_np(rot6: np.ndarray) -> np.ndarray:
    """
    Convert rot6d to rotation matrices.

    rot6: [N, 6] where each row is [R00, R01, R10, R11, R20, R21].
    out : [N, 3, 3]
    """
    _require_2d(rot6, "rot6")
    if rot6.shape[1] != 6:
        raise ValueError(f"rot6 must have D=6, got shape {rot6.shape}")

    rot_cols = rot6.reshape((rot6.shape[0], 3, 2))
    a1 = rot_cols[:, :, 0]
    a2 = rot_cols[:, :, 1]

    eps = 1e-8
    b1_n = np.linalg.norm(a1, axis=1, keepdims=True)
    b1 = a1 / np.maximum(b1_n, eps)

    proj = np.sum(a2 * b1, axis=1, keepdims=True) * b1
    b2_t = a2 - proj
    b2_n = np.linalg.norm(b2_t, axis=1, keepdims=True)
    b2 = b2_t / np.maximum(b2_n, eps)

    b3 = np.cross(b1, b2, axis=1)
    return np.stack([b1, b2, b3], axis=2).astype(np.float32)


def matrix_to_rot6d_np(r: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to row-major 6D columns format.

    r   : [N, 3, 3]
    out : [N, 6] as [R00, R01, R10, R11, R20, R21]
    """
    if r.ndim != 3 or r.shape[1:] != (3, 3):
        raise ValueError(f"r must be [N,3,3], got shape {r.shape}")
    return np.stack([r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1], r[:, 2, 0], r[:, 2, 1]], axis=1).astype(
        np.float32
    )


def to_relative_rot6d_eepose20(a_abs: np.ndarray) -> np.ndarray:
    """
    Convert absolute eepose_6drot actions to relative-rotation eepose_6drot.

    a_abs: [T, 20]
      left : xyz(0:3), rot6d(3:9), grip(9)
      right: xyz(10:13), rot6d(13:19), grip(19)

    output has the same shape. Positions/grippers are unchanged.
    Rotations become relative to t=0 for each arm:
      R_rel[t] = R0^T @ R[t]
    """
    _require_2d(a_abs, "a_abs")
    if a_abs.shape[1] != 20:
        raise ValueError(f"eepose_6drot actions must be [T,20], got {a_abs.shape}")

    out = np.asarray(a_abs, dtype=np.float32).copy()
    idx = [(3, 9), (13, 19)]
    for s, e in idx:
        r_abs = rot6d_to_matrix_np(out[:, s:e])
        r0_t = r_abs[0].T
        r_rel = np.einsum("ij,tjk->tik", r0_t, r_abs)
        out[:, s:e] = matrix_to_rot6d_np(r_rel)
    return out


def from_relative_rot6d_eepose20(a_rel: np.ndarray, ref_abs: np.ndarray) -> np.ndarray:
    """Inverse of to_relative_rot6d_eepose20 for deployment: R[t] = R0 @ R_rel[t], with R0 the rotation of
    ref_abs [20], the robot's current absolute eepose_6drot pose. Positions/grippers pass through unchanged."""
    _require_2d(a_rel, "a_rel")
    _require_1d(ref_abs, "ref_abs")
    if a_rel.shape[1] != 20 or ref_abs.shape[0] != 20:
        raise ValueError(f"eepose_6drot actions must be [T,20] and ref [20], got {a_rel.shape} and {ref_abs.shape}")

    out = np.asarray(a_rel, dtype=np.float32).copy()
    ref = np.asarray(ref_abs, dtype=np.float32)
    for s, e in [(3, 9), (13, 19)]:
        r0 = rot6d_to_matrix_np(ref[None, s:e])[0]
        r_abs = np.einsum("ij,tjk->tik", r0, rot6d_to_matrix_np(out[:, s:e]))
        out[:, s:e] = matrix_to_rot6d_np(r_abs)
    return out


def _interp_actions_at_times(a_src: np.ndarray, ts_src_s: np.ndarray, ts_q_s: np.ndarray, interp_mode: str) -> np.ndarray:
    _require_2d(a_src, "a_src")
    _require_1d(ts_src_s, "ts_src_s")
    _require_1d(ts_q_s, "ts_q_s")
    _check_interp_mode(interp_mode)
    if a_src.shape[0] != ts_src_s.shape[0]:
        raise ValueError(f"a_src and ts_src_s length mismatch: {a_src.shape[0]} vs {ts_src_s.shape[0]}")
    _require_monotonic_increasing(ts_src_s, "ts_src_s")

    lo = float(ts_src_s[0])
    hi = float(ts_src_s[ts_src_s.shape[0] - 1])
    if np.any(ts_q_s < lo) or np.any(ts_q_s > hi):
        raise ValueError(
            f"Query timestamps outside source range: query=[{float(np.min(ts_q_s)):.6f},{float(np.max(ts_q_s)):.6f}] "
            f"source=[{lo:.6f},{hi:.6f}]"
        )

    if interp_mode == "linear":
        out = np.zeros((ts_q_s.shape[0], a_src.shape[1]), dtype=np.float32)
        for d in range(a_src.shape[1]):
            out[:, d] = np.interp(ts_q_s, ts_src_s, a_src[:, d]).astype(np.float32)
        return out

    if interp_mode == "cubic":
        from scipy.interpolate import CubicSpline

        # Degenerate one-sample sequence: cubic spline is undefined.
        # In this case, the only valid interpolation is the constant value.
        if ts_src_s.shape[0] == 1:
            return np.repeat(a_src[0:1], repeats=ts_q_s.shape[0], axis=0).astype(np.float32)
        cs = CubicSpline(ts_src_s.astype(np.float64), a_src.astype(np.float64), axis=0)
        return cs(ts_q_s.astype(np.float64)).astype(np.float32)

    # nearest
    idx_r = np.searchsorted(ts_src_s, ts_q_s, side="left")
    idx_r = np.clip(idx_r, 0, ts_src_s.shape[0] - 1)
    idx_l = np.maximum(idx_r - 1, 0)
    dl = np.abs(ts_q_s - ts_src_s[idx_l])
    dr = np.abs(ts_src_s[idx_r] - ts_q_s)
    idx = np.where(dr < dl, idx_r, idx_l)
    return a_src[idx].astype(np.float32)


def sync_actions_to_fps(
    a_src: np.ndarray,
    ts_src_s: np.ndarray,
    fps_dst: int,
    interp_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample actions to a fixed fps timeline.

    a_src   : [N, D]
    ts_src_s: [N] seconds, strictly increasing
    fps_dst : destination fps (e.g. 30)
    out     : (a_dst, ts_dst_s)
      a_dst   [M, D]
      ts_dst_s[M] uniform timeline in seconds
    """
    _require_2d(a_src, "a_src")
    _require_1d(ts_src_s, "ts_src_s")
    _check_interp_mode(interp_mode)
    if a_src.shape[0] != ts_src_s.shape[0]:
        raise ValueError(f"a_src and ts_src_s length mismatch: {a_src.shape[0]} vs {ts_src_s.shape[0]}")
    if fps_dst <= 0:
        raise ValueError(f"fps_dst must be > 0, got {fps_dst}")
    _require_monotonic_increasing(ts_src_s, "ts_src_s")

    t0 = float(ts_src_s[0])
    t1 = float(ts_src_s[ts_src_s.shape[0] - 1])
    dt = 1.0 / float(fps_dst)
    n = int(np.floor((t1 - t0) / dt)) + 1
    if n <= 0:
        raise ValueError(f"Invalid destination timeline length n={n} from t0={t0:.6f}, t1={t1:.6f}, fps={fps_dst}")

    ts_dst = t0 + np.arange(n, dtype=np.float64) * float(dt)
    ts_dst = ts_dst[ts_dst <= (t1 + 1e-9)]
    if ts_dst.shape[0] <= 0 or ts_dst[ts_dst.shape[0] - 1] < t1:
        ts_dst = np.concatenate([ts_dst, np.array([t1], dtype=np.float64)], axis=0)

    ts_q = ts_dst.astype(np.float32)
    if ts_q.shape[0] > 1:
        keep = np.ones(ts_q.shape[0], dtype=bool)
        keep[1:] = ts_q[1:] > ts_q[:-1]
        ts_q = ts_q[keep]
    if ts_q.shape[0] <= 0:
        raise ValueError("Destination timeline collapsed to zero length after float32 dedup")
    if ts_q[ts_q.shape[0] - 1] < np.float32(t1):
        ts_q = np.concatenate([ts_q, np.array([np.float32(t1)], dtype=np.float32)], axis=0)

    a_dst = _interp_actions_at_times(
        a_src=np.asarray(a_src, dtype=np.float32),
        ts_src_s=np.asarray(ts_src_s, dtype=np.float32),
        ts_q_s=ts_q,
        interp_mode=interp_mode,
    )
    return a_dst.astype(np.float32), ts_q.astype(np.float32)


def sample_actions_at_times(
    a_src: np.ndarray,
    ts_src_s: np.ndarray,
    ts_q_s: np.ndarray,
    interp_mode: str,
) -> np.ndarray:
    """
    Sample action sequence at query timestamps.

    a_src   : [N, D]
    ts_src_s: [N] seconds
    ts_q_s  : [T] query times in seconds
    out     : [T, D]
    """
    return _interp_actions_at_times(
        a_src=np.asarray(a_src, dtype=np.float32),
        ts_src_s=np.asarray(ts_src_s, dtype=np.float32),
        ts_q_s=np.asarray(ts_q_s, dtype=np.float32),
        interp_mode=interp_mode,
    )

