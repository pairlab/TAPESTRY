"""Action-normalization statistics for a directory of HDF5 demos (layout: tapestry/loaders/hdf5_demo_dataset.py).

    python scripts/compute_normalizer_stats.py --demo-dir data/my_robot --action-space eepose_6drot \
        --horizon 16 --output data/my_robot/normalizer_stats.json [--relative-rotation-actions]

Point training.dataset.normalizer_stats_path at the output (see cfg/finetune.yaml). --horizon must equal
training.true_horizon and --relative-rotation-actions must match training.dataset.relative_rotation_actions.
"""
import argparse
import glob
import os

import h5py
import numpy as np

from tapestry.utils.action_normalizer import ACTION_SPACE_CONFIGS, compute_normalizer_stats
from tapestry.utils.action_preprocess import to_relative_rot6d_eepose20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-dir", required=True, help="directory of *.hdf5 demos")
    ap.add_argument("--action-space", required=True, choices=sorted(ACTION_SPACE_CONFIGS), help="actions/<action_space> dataset")
    ap.add_argument("--horizon", type=int, required=True, help="action chunk length (training.true_horizon)")
    ap.add_argument("--output", required=True, help="stats JSON to write (training.dataset.normalizer_stats_path)")
    ap.add_argument("--relative-rotation-actions", action="store_true", help="rot6d relative to the chunk start (eepose_6drot only)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.demo_dir), "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no *.hdf5 files in {args.demo_dir}")
    key = f"actions/{args.action_space}"
    seqs = []
    for path in files:
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"{path} has no dataset {key!r}")
            seqs.append(f[key][:].astype(np.float32))
    if args.relative_rotation_actions:
        if args.action_space != "eepose_6drot":
            raise ValueError("--relative-rotation-actions is only defined for eepose_6drot")
        seqs = [to_relative_rot6d_eepose20(seq) for seq in seqs]
    print(f"{len(seqs)} demos, {sum(len(s) for s in seqs)} steps, action dim {seqs[0].shape[1]}")

    normalizer = compute_normalizer_stats(seqs, args.action_space, args.horizon)
    normalizer.save(args.output, num_demos=len(seqs))

    chunk = seqs[0][: args.horizon]  # roundtrip check on values that are not clipped by the normalizer
    normed = normalizer.normalize(chunk)
    err = np.abs(normalizer.denormalize(normed, chunk[0]) - chunk)[np.abs(normed) < 1.5]
    if err.size and err.max() > 1e-4:
        raise RuntimeError(f"normalize/denormalize roundtrip error {err.max():.2e}")
    print(f"roundtrip ok; normalized dims {normalizer.normalize_dims}, delta_actions={normalizer.delta_actions}")


if __name__ == "__main__":
    main()
