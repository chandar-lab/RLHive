import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {
    "atari": ["atari_py~=0.2.0", "opencv-python>=4."],
    "minatar": [
        "MinAtar @ https://github.com/kenjyoung/MinAtar/archive/8b39a18a60248ede15ce70142b557f3897c4e1eb.zip"
    ],
    "marlgrid": [
        "marlgrid @ https://github.com/kandouss/marlgrid/archive/e88c40bad07653575ac11fe2f3a115e4de3d13e9.zip"
    ],
    "test": ["pytest", "pytest-lazy-fixture"],
}

setuptools.setup(
    name="rl-hive",
    version="0.1.0",
    description="A package to support RL research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chandar-lab/RLHive",
    project_urls={
        "Bug Tracker": "https://github.com/chandar-lab/RLHive/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    install_requires=[
        "gym>=0.18.0",
        "numpy>=1.18.0",
        "PyYAML>=5.1",
        "torch>=1.8.0",
        "wandb~=0.10.30",
    ],
    extras_require=extras,
)
