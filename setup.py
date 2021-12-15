#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""This file is for distribute and install sierras"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

from setuptools import find_packages, setup

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

REQUIREMENTS = [
    "matplotlib",
    "numpy",
    "pandas",
    "pint",
    "scikit-learn",
    "uncertainties",
]

with open(PATH / "sierras" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="sierras",
    version=VERSION,
    description="Perform arrhenius plots for diffusion coefficients",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Francisco Fernandez",
    author_email="fernandezfrancisco2195@gmail.com",
    url="https://github.com/fernandezfran/sierras",
    packages=find_packages(),
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=["sierras", "arrhenius plot", "diffusion coeficient"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
