import os

import setuptools

with open("version.txt") as f:
    VERSION = f.read().strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

atari = ["gymnasium[atari,accept-rom-license]~=0.27.0", "opencv-python~=4.0"]
minigrid = ["minigrid~=2.1.0"]
minatar = ["MinAtar>=1.0.10"]
petting_zoo = ["pettingzoo[sisl,atari,classic]~=1.22"]
visualization = ["pandas~=1.0"]
test = ["pytest~=6.2", "pytest-lazy-fixture~=0.6"]


setuptools.setup(
    name="rlhive",
    version=VERSION,
    description="A package to support RL research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chandar-lab/RLHive",
    project_urls={
        "Bug Tracker": "https://github.com/chandar-lab/RLHive/issues",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"hive": ["configs/**.yml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    install_requires=[
        "gymnasium~=0.27.0",
        "numpy~=1.18",
        "PyYAML~=5.4",
        "torch>=1.8",
        "wandb>=0.12.18",
        "matplotlib~=3.0",
    ],
    extras_require={
        "atari": atari,
        "minigrid": minigrid,
        "minatar": minatar,
        "petting_zoo": petting_zoo,
        "test": test,
        "all": atari + minigrid + minatar + petting_zoo + test,
    },
    entry_points={
        "console_scripts": [
            "hive_main = hive.main:main",
        ]
    },
)
