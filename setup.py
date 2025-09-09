# setup.py

from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="nstm",
    version="0.1.0",
    author="NSTM Team",
    author_email="contact@nstm.ai",
    description="Neural State Transition Machine - A novel approach to sequence modeling with explicit state management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nstm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "black>=21.5b1",
            "flake8>=3.9.2",
        ],
        "test": [
            "pytest>=6.2.4",
        ],
    },
)