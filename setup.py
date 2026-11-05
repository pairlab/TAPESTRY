"""Setup script for TAPESTRY."""

from pathlib import Path

from setuptools import find_packages, setup

requirements = [line.split("#", 1)[0].strip()
                for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()]

setup(
    name="tapestry",
    version="1.0.0",
    description="TAPESTRY: A Single Backbone for Video Generation and Robot Control",
    license="MIT",
    packages=find_packages(include=["tapestry", "tapestry.*"]),
    install_requires=[line for line in requirements if line],
    extras_require={"test": ["pytest"], "decord": ["decord==0.6.0"]},
    python_requires=">=3.10",
)
