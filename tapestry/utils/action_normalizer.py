"""
Percentile-based action normalization.

Implements per-timestep, per-feature percentile-based normalization:
- Normalizes position dimensions using 2nd/98th percentiles
- Maps [p02, p98] -> [-1, 1], clips outliers to [-1.5, 1.5]
- Skips rotation and gripper dimensions

Two modes:
- Absolute actions (delta_actions=False): subtracts action[0] to make actions relative
  to the chunk start before scaling. Used for absolute EE pose action spaces (e.g., the bimanual eepose spaces).
- Delta actions (delta_actions=True): Direct percentile scaling without reference subtraction.
  Used for action spaces that are already deltas (e.g., LIBERO).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch


# Action dimension structure for bimanual robot
#
# eepose_6drot (20D):
#   Left arm:  [pos(3), rot6d(6), gripper(1)] = dims 0-9
#   Right arm: [pos(3), rot6d(6), gripper(1)] = dims 10-19
#   Normalize: position only (0-2, 10-12)
#   Skip: 6D rotation (3-8, 13-18) - already in [-1, 1] range
#   Skip: gripper (9, 19) - binary values (0 or 1)
#
# eepose_ypr (14D):
#   Left arm:  [pos(3), ypr(3), gripper(1)] = dims 0-6
#   Right arm: [pos(3), ypr(3), gripper(1)] = dims 7-13
#   Normalize: position only (0-2, 7-9)
#   Skip: yaw/pitch/roll (3-5, 10-12) - angular values
#   Skip: gripper (6, 13) - binary values (0 or 1)
#
# Action dimension structure for LIBERO (single arm)
#
# libero (7D):
#   [pos(3), axis_angle(3), gripper(1)]
#   Normalize: position only (0-2)
#   Skip: axis-angle rotation (3-5) - angular values
#   Skip: gripper (6) - binary value (0 or 1)
#
ACTION_SPACE_CONFIGS = {
    "eepose_6drot": {
        "total_dim": 20,
        "normalize_dims": [0, 1, 2, 10, 11, 12],  # Position only (gripper is binary, skip)
        "delta_actions": False,  # Absolute EE poses
    },
    "eepose_ypr": {
        "total_dim": 14,
        "normalize_dims": [0, 1, 2, 7, 8, 9],  # Position only (gripper is binary, skip)
        "delta_actions": False,  # Absolute EE poses
    },
    "joints": {
        "total_dim": 14,
        "normalize_dims": list(range(14)),  # All joints
        "delta_actions": False,  # Absolute joint positions
    },
    "libero": {
        "total_dim": 7,
        "normalize_dims": [0, 1, 2],  # Position only (axis-angle and gripper skip)
        "delta_actions": True,  # Already delta actions (delta_xyz, delta_axisangle, gripper)
    },
}


class ActionNormalizer:
    """Percentile-based action normalizer.

    Two modes controlled by delta_actions:
    - delta_actions=False: Subtracts action[0] before normalizing,
      adds ref_action back during denormalization. For absolute EE pose action spaces.
    - delta_actions=True: Direct percentile scaling without reference subtraction.
      For action spaces that are already deltas (e.g., LIBERO).

    Only normalizes specified dimensions (typically XYZ positions) using per-timestep
    2nd/98th percentile statistics. Other dimensions pass through unchanged.

    Formula: y = clip(2 * (x - p02) / (p98 - p02) - 1, -1.5, 1.5)
    """

    def __init__(
        self,
        p02: np.ndarray,
        p98: np.ndarray,
        normalize_dims: List[int],
        action_space: str,
        horizon: int,
        delta_actions: bool,
    ):
        """Initialize normalizer with precomputed statistics.

        Args:
            p02: 2nd percentile stats, shape (horizon, num_normalize_dims)
            p98: 98th percentile stats, shape (horizon, num_normalize_dims)
            normalize_dims: List of dimension indices to normalize
            action_space: Action space name (e.g., 'eepose_6drot', 'libero')
            horizon: Action prediction horizon (number of timesteps)
            delta_actions: If True, actions are already deltas and the reference subtraction is skipped
        """
        self.p02 = np.array(p02, dtype=np.float32)
        self.p98 = np.array(p98, dtype=np.float32)
        self.normalize_dims = normalize_dims
        self.action_space = action_space
        self.horizon = horizon
        self.delta_actions = delta_actions

        for name, stats in (("p02", self.p02), ("p98", self.p98)):
            if stats.shape != (horizon, len(normalize_dims)):
                raise ValueError(f"{name} shape mismatch: {stats.shape} vs ({horizon}, {len(normalize_dims)})")

        # Precompute range for efficiency
        self.range = self.p98 - self.p02
        # Avoid division by zero - use small epsilon
        self.range = np.where(self.range < 1e-8, 1e-8, self.range)

    @classmethod
    def from_dict(cls, data: Dict) -> "ActionNormalizer":
        """Inverse of to_dict (also the content of a stats JSON file)."""
        return cls(
            p02=np.array(data["stats"]["p02"]),
            p98=np.array(data["stats"]["p98"]),
            normalize_dims=data["normalize_dims"],
            action_space=data["action_space"],
            horizon=data["horizon"],
            delta_actions=data["delta_actions"],
        )

    @classmethod
    def from_stats_file(cls, stats_path: Union[str, Path]) -> "ActionNormalizer":
        """Load normalizer from a stats JSON file (scripts/compute_normalizer_stats.py)."""
        with open(stats_path, "r") as f:
            return cls.from_dict(json.load(f))

    def normalize(self, actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize actions using percentile scaling.

        For absolute actions (delta_actions=False):
          1. Convert to relative at normalize_dims: x = act[t, dim] - act[0, dim]
          2. Scale: y = clip(2 * (x - p02) / (p98 - p02) - 1, -1.5, 1.5)

        For delta actions (delta_actions=True):
          1. Scale directly: y = clip(2 * (x - p02) / (p98 - p02) - 1, -1.5, 1.5)

        Args:
            actions: Action tensor, shape (T, D) or (B, T, D)

        Returns:
            Normalized actions, same shape as input
        """
        is_torch = isinstance(actions, torch.Tensor)
        device = actions.device if is_torch else None

        if is_torch:
            act = actions.cpu().numpy().copy()
        else:
            act = actions.copy()

        # Ensure 3D: (B, T, D)
        added_batch = False
        if act.ndim == 2:
            act = act[np.newaxis, :, :]  # (T, D) -> (1, T, D)
            added_batch = True

        B, T, D = act.shape
        if T > self.horizon:
            raise ValueError(f"T={T} exceeds horizon={self.horizon}")

        # For absolute actions, save reference values BEFORE modifying
        if not self.delta_actions:
            ref_values = act[:, 0, :].copy()  # (B, D)

        for t in range(T):
            for i, dim in enumerate(self.normalize_dims):
                if self.delta_actions:
                    x = act[:, t, dim]
                else:
                    # Relative: subtract original action[0] at this dim
                    x = act[:, t, dim] - ref_values[:, dim]
                # Percentile scale
                p02_t = self.p02[t, i]
                rng_t = self.range[t, i]
                act[:, t, dim] = np.clip(2.0 * (x - p02_t) / rng_t - 1.0, -1.5, 1.5)

        if added_batch:
            act = act[0]  # (1, T, D) -> (T, D)

        if is_torch:
            return torch.from_numpy(act).to(device)
        return act

    def denormalize(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        ref_action: Union[np.ndarray, torch.Tensor, None],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize actions (and convert to absolute if not delta).

        For absolute actions (delta_actions=False):
          1. Inverse scale: x = (y + 1) * (p98 - p02) / 2 + p02
          2. Convert to absolute: act[dim] += ref_action[dim]

        For delta actions (delta_actions=True):
          1. Inverse scale only: x = (y + 1) * (p98 - p02) / 2 + p02

        Args:
            actions: Normalized actions, shape (T, D) or (B, T, D)
            ref_action: Reference action (current EE pose), shape (D,).
                       Required when delta_actions=False; None for delta action spaces.

        Returns:
            Denormalized actions, same shape as input
        """
        if not self.delta_actions and ref_action is None:
            raise ValueError("ref_action is required for absolute action spaces (delta_actions=False)")

        is_torch = isinstance(actions, torch.Tensor)
        device = actions.device if is_torch else None

        if is_torch:
            act = actions.cpu().numpy().copy()
        else:
            act = actions.copy()

        # Ensure 3D: (B, T, D)
        added_batch = False
        if act.ndim == 2:
            act = act[np.newaxis, :, :]  # (T, D) -> (1, T, D)
            added_batch = True

        B, T, D = act.shape
        if T > self.horizon:
            raise ValueError(f"T={T} exceeds horizon={self.horizon}")

        # Prepare reference for absolute action spaces
        if not self.delta_actions:
            if isinstance(ref_action, torch.Tensor):
                ref = ref_action.cpu().numpy()
            else:
                ref = np.asarray(ref_action, dtype=np.float32)
            if ref.ndim != 1:
                raise ValueError(f"ref_action must be 1D, got shape {ref.shape}")

        # Denormalize at normalize_dims only
        for t in range(T):
            for i, dim in enumerate(self.normalize_dims):
                y = act[:, t, dim]
                p02_t = self.p02[t, i]
                rng_t = self.range[t, i]
                x = (y + 1.0) * rng_t / 2.0 + p02_t
                if not self.delta_actions:
                    x = x + ref[dim]
                act[:, t, dim] = x

        if added_batch:
            act = act[0]  # (1, T, D) -> (T, D)

        if is_torch:
            return torch.from_numpy(act).to(device)
        return act

    def to_dict(self) -> Dict:
        """Convert normalizer to dictionary for serialization."""
        return {
            "action_space": self.action_space,
            "horizon": self.horizon,
            "normalize_dims": self.normalize_dims,
            "delta_actions": self.delta_actions,
            "stats": {
                "p02": self.p02.tolist(),
                "p98": self.p98.tolist(),
            },
        }

    def save(self, path: Union[str, Path], num_demos: int):
        """Save normalizer stats to JSON file.

        Args:
            path: Output file path
            num_demos: Number of demos used to compute stats (for metadata)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        data["num_demos"] = num_demos
        data["computed_at"] = datetime.now().isoformat()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved normalizer stats to {path}")


def compute_normalizer_stats(
    action_sequences: List[np.ndarray],
    action_space: str,
    horizon: int,
) -> ActionNormalizer:
    """Compute normalizer statistics from action sequences.

    Samples ALL possible chunk starting positions from each demo sequence.

    For absolute action spaces (delta_actions=False):
      Stats are computed on relative actions: action[t] - action[0] for each chunk.

    For delta action spaces (delta_actions=True):
      Stats are computed on raw action values (already deltas).

    Args:
        action_sequences: List of action arrays, each shape (T, D)
        action_space: Action space name (e.g., 'eepose_6drot', 'libero')
        horizon: Action prediction horizon

    Returns:
        ActionNormalizer with computed percentile statistics
    """
    config = ACTION_SPACE_CONFIGS[action_space]
    norm_dims = config["normalize_dims"]
    total_dim = config["total_dim"]
    delta_actions = config["delta_actions"]
    n_dims = len(norm_dims)

    # Collect values: values[timestep][dim_idx] = list of values
    values = [[[] for _ in range(n_dims)] for _ in range(horizon)]

    n_chunks = 0
    for seq in action_sequences:
        if seq.shape[1] != total_dim:
            raise ValueError(f"Action dim {seq.shape[1]} != expected {total_dim}")

        seq_len = seq.shape[0]

        # Sample all possible chunk starting positions
        for start in range(max(1, seq_len - horizon + 1)):
            chunk = seq[start:start + horizon]
            chunk_len = chunk.shape[0]

            for t in range(chunk_len):
                for i, dim in enumerate(norm_dims):
                    if delta_actions:
                        # Delta actions: use raw values
                        values[t][i].append(chunk[t, dim])
                    else:
                        # Absolute actions: relative to chunk start
                        values[t][i].append(chunk[t, dim] - chunk[0, dim])

            n_chunks += 1

    print(f"  Sampled {n_chunks} chunks from {len(action_sequences)} demos")

    # Compute percentiles
    p02 = np.zeros((horizon, n_dims), dtype=np.float32)
    p98 = np.zeros((horizon, n_dims), dtype=np.float32)

    for t in range(horizon):
        for i in range(n_dims):
            if len(values[t][i]) == 0:
                raise ValueError(f"No data for timestep {t}, dim {norm_dims[i]}")
            arr = np.array(values[t][i])
            p02[t, i] = np.percentile(arr, 2)
            p98[t, i] = np.percentile(arr, 98)

    return ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=norm_dims,
        action_space=action_space,
        horizon=horizon,
        delta_actions=delta_actions,
    )
