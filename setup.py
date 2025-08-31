import os
import sys
from setuptools import setup, find_packages
from typing import List

# Define constants
PROJECT_NAME = "VisiTrail"
VERSION = "1.0.0"
AUTHOR = "Abdul Rehman, Ilona Heldal, and Jerry Chun-Wei Lin"
EMAIL = "author@example.com"
DESCRIPTION = "A Cognitive Visualization Tool for Time-Series Analysis of Eye Tracking Data from Attention Game"
LICENSE = "MIT"
URL = "https://github.com/author/VisiTrail"

# Define dependencies
INSTALL_REQUIRES: List[str] = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "opencv-python",
]

# Define development dependencies
EXTRA_REQUIRES: dict = {
    "dev": [
        "pytest",
        "flake8",
        "mypy",
        "black",
    ],
}

# Define package data
PACKAGE_DATA: dict = {
    "": ["*.txt", "*.md", "*.pdf"],
}

# Define entry points
ENTRY_POINTS: dict = {
    "console_scripts": [
        "visitrail=visitrail.main:main",
    ],
}

def read_file(filename: str) -> str:
    """Read the contents of a file."""
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

def main() -> None:
    """Main function to setup the package."""
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description=read_file("README.md"),
        long_description_content_type="text/markdown",
        url=URL,
        license=LICENSE,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        package_data=PACKAGE_DATA,
        entry_points=ENTRY_POINTS,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()