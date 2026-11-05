"""
Tests for ActionNormalizer class.

Tests (direct percentile scaling uses delta_actions=True; the absolute mode
subtracts action[0] and needs ref_action to invert; that mode is covered separately):
1. Normalize/denormalize roundtrip preserves data
2. Normalization maps percentile range to [-1, 1]
3. Outliers are clipped to [-1.5, 1.5]
4. Only position dims are normalized, rotation/gripper pass through
5. Per-timestep statistics work correctly
6. Save/load roundtrip preserves stats
7. Absolute-mode roundtrip with ref_action
8. compute_normalizer_stats end to end

Usage:
    python tests/test_action_normalizer.py

    # Or with pytest
    pytest tests/test_action_normalizer.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from tapestry.utils.action_normalizer import (
    ActionNormalizer,
    compute_normalizer_stats,
)


def test_normalize_denormalize_roundtrip():
    """Test that normalize -> denormalize recovers original data."""
    np.random.seed(42)

    horizon = 16
    action_dim = 20  # eepose_6drot
    normalize_dims = [0, 1, 2, 9, 10, 11, 12, 19]  # Position + gripper dims

    # Create synthetic stats
    p02 = np.random.randn(horizon, len(normalize_dims)).astype(np.float32) - 0.5
    p98 = p02 + np.abs(np.random.randn(horizon, len(normalize_dims))).astype(np.float32) + 0.1

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=True,
    )

    # Create test data within percentile range
    test_actions = np.zeros((horizon, action_dim), dtype=np.float32)
    for t in range(horizon):
        for i, dim in enumerate(normalize_dims):
            # Sample within [p02, p98] range
            test_actions[t, dim] = np.random.uniform(p02[t, i], p98[t, i])
        # Fill other dims with random values (should pass through unchanged)
        for dim in range(action_dim):
            if dim not in normalize_dims:
                test_actions[t, dim] = np.random.randn()

    # Roundtrip
    normalized = normalizer.normalize(test_actions)
    denormalized = normalizer.denormalize(normalized, None)

    # Check roundtrip error
    max_error = np.abs(test_actions - denormalized).max()
    print(f"  Roundtrip max error: {max_error:.2e}")

    assert max_error < 1e-5, f"Roundtrip error too large: {max_error}"

    # Check that non-normalized dims are unchanged
    for dim in range(action_dim):
        if dim not in normalize_dims:
            assert np.allclose(test_actions[:, dim], normalized[:, dim]), \
                f"Non-normalized dim {dim} was modified"

    print("  PASSED: normalize/denormalize roundtrip")


def test_normalization_range():
    """Test that normalization maps [p02, p98] -> [-1, 1]."""
    horizon = 8
    normalize_dims = [0, 1, 2]

    p02 = np.array([[0.0, -1.0, 2.0]] * horizon, dtype=np.float32)
    p98 = np.array([[1.0, 1.0, 4.0]] * horizon, dtype=np.float32)

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=True,
    )

    # Test that p02 maps to -1, p98 maps to 1
    test_actions = np.zeros((horizon, 20), dtype=np.float32)
    for t in range(horizon):
        test_actions[t, 0] = p02[t, 0]  # Should map to -1
        test_actions[t, 1] = p98[t, 1]  # Should map to 1
        test_actions[t, 2] = (p02[t, 2] + p98[t, 2]) / 2  # Should map to 0

    normalized = normalizer.normalize(test_actions)

    # Check values at percentile boundaries
    assert np.allclose(normalized[:, 0], -1.0), f"p02 should map to -1, got {normalized[0, 0]}"
    assert np.allclose(normalized[:, 1], 1.0), f"p98 should map to 1, got {normalized[0, 1]}"
    assert np.allclose(normalized[:, 2], 0.0), f"midpoint should map to 0, got {normalized[0, 2]}"

    print("  PASSED: normalization range mapping")


def test_outlier_clipping():
    """Test that outliers are clipped to [-1.5, 1.5]."""
    horizon = 4
    normalize_dims = [0]

    p02 = np.array([[0.0]] * horizon, dtype=np.float32)
    p98 = np.array([[1.0]] * horizon, dtype=np.float32)

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=True,
    )

    # Create data with outliers
    test_actions = np.zeros((horizon, 20), dtype=np.float32)
    test_actions[0, 0] = -0.5  # Below p02 -> maps to -2, clipped to -1.5
    test_actions[1, 0] = 1.5   # Above p98 -> maps to 2, clipped to 1.5
    test_actions[2, 0] = 0.5   # Within range -> maps to 0

    normalized = normalizer.normalize(test_actions)

    assert normalized[0, 0] == -1.5, f"Low outlier should clip to -1.5, got {normalized[0, 0]}"
    assert normalized[1, 0] == 1.5, f"High outlier should clip to 1.5, got {normalized[1, 0]}"
    assert normalized[2, 0] == 0.0, f"Midpoint should be 0, got {normalized[2, 0]}"

    print("  PASSED: outlier clipping")


def test_per_timestep_stats():
    """Test that per-timestep statistics are applied correctly."""
    horizon = 4
    normalize_dims = [0]

    # Different stats for each timestep
    p02 = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    p98 = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=True,
    )

    # Each timestep should normalize differently
    test_actions = np.zeros((horizon, 20), dtype=np.float32)
    for t in range(horizon):
        test_actions[t, 0] = p02[t, 0]  # Each maps to -1

    normalized = normalizer.normalize(test_actions)

    # All should map to -1 despite different input values
    assert np.allclose(normalized[:, 0], -1.0), \
        f"All p02 values should map to -1, got {normalized[:, 0]}"

    print("  PASSED: per-timestep statistics")


def test_torch_tensor_support():
    """Test that torch tensors are handled correctly."""
    horizon = 8
    normalize_dims = [0, 1, 2]

    p02 = np.zeros((horizon, len(normalize_dims)), dtype=np.float32)
    p98 = np.ones((horizon, len(normalize_dims)), dtype=np.float32)

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=True,
    )

    # Test with torch tensor; keep normalize dims inside [p02, p98] = [0, 1]
    # so clipping (by design) cannot break the roundtrip
    test_actions = torch.randn(horizon, 20)
    test_actions[:, :3] = torch.rand(horizon, 3)
    normalized = normalizer.normalize(test_actions)
    denormalized = normalizer.denormalize(normalized, None)

    assert isinstance(normalized, torch.Tensor), "Output should be torch tensor"
    assert isinstance(denormalized, torch.Tensor), "Output should be torch tensor"

    # Check roundtrip
    max_error = torch.abs(test_actions - denormalized).max().item()
    assert max_error < 1e-5, f"Torch roundtrip error too large: {max_error}"

    print("  PASSED: torch tensor support")


def test_batch_dimension():
    """Test that batch dimension is handled correctly."""
    horizon = 8
    normalize_dims = [0, 1, 2]
    batch_size = 4

    p02 = np.zeros((horizon, len(normalize_dims)), dtype=np.float32)
    p98 = np.ones((horizon, len(normalize_dims)), dtype=np.float32)

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=True,
    )

    # Test with batched input - use values within [p02, p98] range to avoid clipping
    # (clipping breaks roundtrip by design)
    test_actions = np.zeros((batch_size, horizon, 20), dtype=np.float32)
    for b in range(batch_size):
        for t in range(horizon):
            for i, dim in enumerate(normalize_dims):
                # Sample within [p02, p98] range
                test_actions[b, t, dim] = np.random.uniform(p02[t, i], p98[t, i])
            # Fill other dims with random values (pass through unchanged)
            for dim in range(20):
                if dim not in normalize_dims:
                    test_actions[b, t, dim] = np.random.randn()

    normalized = normalizer.normalize(test_actions)
    denormalized = normalizer.denormalize(normalized, None)

    assert normalized.shape == (batch_size, horizon, 20), \
        f"Wrong output shape: {normalized.shape}"

    max_error = np.abs(test_actions - denormalized).max()
    assert max_error < 1e-5, f"Batch roundtrip error too large: {max_error}"

    print("  PASSED: batch dimension handling")


def test_save_load_roundtrip():
    """Test that save/load preserves normalizer stats."""
    horizon = 8
    normalize_dims = [0, 1, 2, 9, 10, 11, 12, 19]  # Position + gripper

    np.random.seed(123)
    p02 = np.random.randn(horizon, len(normalize_dims)).astype(np.float32)
    p98 = p02 + np.abs(np.random.randn(horizon, len(normalize_dims))).astype(np.float32) + 0.1

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=True,
    )

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        normalizer.save(temp_path, num_demos=100)
        loaded = ActionNormalizer.from_stats_file(temp_path)

        # Check that stats match
        assert np.allclose(normalizer.p02, loaded.p02), "p02 mismatch after save/load"
        assert np.allclose(normalizer.p98, loaded.p98), "p98 mismatch after save/load"
        assert normalizer.normalize_dims == loaded.normalize_dims, "normalize_dims mismatch"
        assert normalizer.action_space == loaded.action_space, "action_space mismatch"
        assert normalizer.horizon == loaded.horizon, "horizon mismatch"

        # Test that loaded normalizer works correctly
        test_actions = np.random.randn(horizon, 20).astype(np.float32)
        norm1 = normalizer.normalize(test_actions)
        norm2 = loaded.normalize(test_actions)

        assert np.allclose(norm1, norm2), "Loaded normalizer gives different results"

        print("  PASSED: save/load roundtrip")
    finally:
        Path(temp_path).unlink()


def test_absolute_mode_roundtrip():
    """Absolute mode: normalize subtracts action[0]; denormalize needs ref_action."""
    np.random.seed(7)

    horizon = 16
    action_dim = 20
    normalize_dims = [0, 1, 2, 9, 10, 11, 12, 19]

    p02 = -np.abs(np.random.randn(horizon, len(normalize_dims))).astype(np.float32) - 0.1
    p98 = np.abs(np.random.randn(horizon, len(normalize_dims))).astype(np.float32) + 0.1

    normalizer = ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=normalize_dims,
        action_space="eepose_6drot",
        horizon=horizon,
        delta_actions=False,
    )
    assert not normalizer.delta_actions

    # Absolute trajectories whose relative values stay within [p02, p98]
    test_actions = np.random.randn(horizon, action_dim).astype(np.float32)
    for t in range(horizon):
        for i, dim in enumerate(normalize_dims):
            rel = np.random.uniform(p02[t, i], p98[t, i])
            test_actions[t, dim] = test_actions[0, dim] if t == 0 else test_actions[0, dim] + rel

    ref = test_actions[0].copy()
    normalized = normalizer.normalize(test_actions)
    denormalized = normalizer.denormalize(normalized, ref_action=ref)

    max_error = np.abs(test_actions - denormalized).max()
    assert max_error < 1e-4, f"Absolute-mode roundtrip error too large: {max_error}"

    print("  PASSED: absolute-mode roundtrip with ref_action")


def test_compute_normalizer_stats():
    """Test the compute_normalizer_stats function."""
    np.random.seed(456)

    horizon = 8
    action_dim = 20
    num_sequences = 50
    num_normalize_dims = 6  # Positions only (gripper is binary, excluded)

    # Generate synthetic action sequences with known distribution
    action_sequences = []
    for _ in range(num_sequences):
        seq_len = np.random.randint(horizon, horizon + 10)
        # Generate actions where position dims have a specific distribution
        seq = np.random.randn(seq_len, action_dim).astype(np.float32)
        action_sequences.append(seq)

    # Compute stats
    normalizer = compute_normalizer_stats(
        action_sequences=action_sequences,
        action_space="eepose_6drot",
        horizon=horizon,
    )

    # Verify stats are reasonable
    assert normalizer.p02.shape == (horizon, num_normalize_dims), f"Wrong p02 shape: {normalizer.p02.shape}"
    assert normalizer.p98.shape == (horizon, num_normalize_dims), f"Wrong p98 shape: {normalizer.p98.shape}"

    # p98 >= p02 everywhere; strictly greater after t=0 (relative actions are
    # identically zero at t=0, so both percentiles are 0 there)
    assert np.all(normalizer.p98 >= normalizer.p02), "p98 should be >= p02"
    assert np.all(normalizer.p98[1:] > normalizer.p02[1:]), "p98 should be > p02 for t>0"

    # Test that normalization works
    test_seq = action_sequences[0][:horizon]
    normalized = normalizer.normalize(test_seq)

    # Most values should be in [-1.5, 1.5] range
    for dim in normalizer.normalize_dims:
        assert np.all(normalized[:, dim] >= -1.5) and np.all(normalized[:, dim] <= 1.5), \
            f"Normalized values out of range for dim {dim}"

    print("  PASSED: compute_normalizer_stats")
