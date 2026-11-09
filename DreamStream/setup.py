from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="dreamstream",
    version="0.1.0",
    description="DreamStream: layout-conditioned streaming video generation for closed-loop driving simulation",
    packages=find_packages(include=["dreamstream", "dreamstream.*", "stream_package", "stream_package.*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
)
