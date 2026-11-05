import numpy as np

from tapestry.utils.action_preprocess import (
    matrix_to_rot6d_np,
    sample_actions_at_times,
    sync_actions_to_fps,
    to_relative_rot6d_eepose20,
)


def _rotz(th: float) -> np.ndarray:
    c = np.cos(th)
    s = np.sin(th)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_to_relative_rot6d_eepose20_t0_identity() -> None:
    t_len = 5
    a = np.zeros((t_len, 20), dtype=np.float32)
    r_l = np.stack([_rotz(0.2 + 0.1 * i) for i in range(t_len)], axis=0)
    r_r = np.stack([_rotz(-0.3 + 0.05 * i) for i in range(t_len)], axis=0)
    a[:, 3:9] = matrix_to_rot6d_np(r_l)
    a[:, 13:19] = matrix_to_rot6d_np(r_r)

    a_rel = to_relative_rot6d_eepose20(a)
    id6 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    assert np.allclose(a_rel[0, 3:9], id6, atol=1e-5)
    assert np.allclose(a_rel[0, 13:19], id6, atol=1e-5)


def test_sync_actions_to_fps_linear() -> None:
    ts = np.array([0.0, 0.2, 0.4], dtype=np.float32)
    a = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )
    a_sync, ts_sync = sync_actions_to_fps(a_src=a, ts_src_s=ts, fps_dst=10, interp_mode="linear")
    # t=0.1 is midpoint between first two source samples.
    idx = int(np.argmin(np.abs(ts_sync - 0.1)))
    assert np.allclose(a_sync[idx], np.array([1.0, 2.0], dtype=np.float32), atol=1e-5)


def test_sync_actions_to_fps_cubic() -> None:
    ts = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    a = np.stack([ts**2, ts**3], axis=1).astype(np.float32)
    a_sync, ts_sync = sync_actions_to_fps(a_src=a, ts_src_s=ts, fps_dst=30, interp_mode="cubic")
    assert a_sync.ndim == 2 and a_sync.shape[1] == 2
    # Cubic spline should remain finite and hit endpoints exactly.
    assert np.all(np.isfinite(a_sync))
    assert np.allclose(a_sync[0], a[0], atol=1e-6)
    assert np.allclose(a_sync[-1], a[-1], atol=1e-6)


def test_sample_actions_at_times_nearest() -> None:
    ts = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    a = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    q = np.array([0.049, 0.151], dtype=np.float32)
    out = sample_actions_at_times(a_src=a, ts_src_s=ts, ts_q_s=q, interp_mode="nearest")
    assert np.allclose(out[:, 0], np.array([0.0, 2.0], dtype=np.float32))


def test_relative_rot6d_roundtrip():
    from tapestry.utils.action_preprocess import from_relative_rot6d_eepose20, matrix_to_rot6d_np, to_relative_rot6d_eepose20

    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 20)).astype(np.float32)
    for s, e in [(3, 9), (13, 19)]:  # valid rotations in the rot6d slots
        q, _ = np.linalg.qr(rng.standard_normal((16, 3, 3)))
        a[:, s:e] = matrix_to_rot6d_np(q.astype(np.float32))
    rel = to_relative_rot6d_eepose20(a)
    back = from_relative_rot6d_eepose20(rel, a[0])
    assert np.allclose(back, a, atol=1e-4)
