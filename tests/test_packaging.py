"""The installed distribution must work without the source checkout on sys.path."""
import os
import runpy
import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_py import build_py


def test_installed_package(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    runpy.run_path(str(root / "setup.py"))

    dist = setuptools.Distribution(metadata)
    dist.script_name = "setup.py"
    cmd = build_py(dist)
    cmd.build_lib = str(tmp_path)
    cmd.ensure_finalized()
    cmd.run()

    # Every module of the source package is installed, and nothing lands at the top level besides tapestry.
    src = {p.relative_to(root) for p in root.joinpath("tapestry").rglob("*.py")}
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*.py")} == src
    assert [p.name for p in tmp_path.iterdir()] == ["tapestry"]
    assert "diffusers==0.34.0" in metadata["install_requires"]
    assert metadata["extras_require"]["test"] == ["pytest"]
    assert metadata["extras_require"]["decord"] == ["decord==0.6.0"]
    assert not any(dep.startswith("decord") for dep in metadata["install_requires"])

    code = ("import tapestry.models.policy, tapestry.losses, tapestry.eval_utils, tapestry.utils.parallel_eval, "
            "tapestry.loaders.hdf5_demo_dataset; print(tapestry.models.policy.__file__)")
    env = dict(os.environ, PYTHONPATH=str(tmp_path), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    p = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=env, capture_output=True, text=True, check=True)
    assert p.stdout.strip() == str(tmp_path / "tapestry" / "models" / "policy.py")
