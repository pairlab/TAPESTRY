"""Write the task description into every HDF5 demo of a directory (the `text` dataset HDF5DemoDataset reads).

    python scripts/add_text_to_hdf5.py --demo-dir data/my_robot --text "pick up the cup"
"""
import argparse
import glob
import os

import h5py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-dir", required=True, help="directory of *.hdf5 demos")
    ap.add_argument("--text", required=True, help="task description")
    ap.add_argument("--overwrite", action="store_true", help="explicitly replace existing task descriptions")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.demo_dir), "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no *.hdf5 files in {args.demo_dir}")
    # Preflight the whole directory before modifying any demo.
    if not args.overwrite:
        for path in files:
            with h5py.File(path, "r") as f:
                if "text" in f:
                    raise FileExistsError(f"{path} already has task text; use --overwrite to replace it")
    for path in files:
        with h5py.File(path, "r+") as f:
            if "text" in f:
                del f["text"]
            f.create_dataset("text", data=args.text, dtype=h5py.string_dtype(encoding="utf-8"))
    print(f"wrote text={args.text!r} to {len(files)} files in {args.demo_dir}")


if __name__ == "__main__":
    main()
