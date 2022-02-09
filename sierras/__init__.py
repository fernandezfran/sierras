#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

r"""A module to perform arrhenius plots for diffusion coefficients.

It is well known that the diffusion coefficients follows an Arrhenius-like
relation with temperature, i.e.

.. math::
    D = D_0 e^{-E / R T}

where :math:`E` is the activation energy of the diffusive process, :math:`R`
the universal gas constant and :math:`D_0` is a pre-exponential factor.

Taking the natural logarithm of this equation yields to a linear relationship

.. math::
    \ln D = - \frac{E}{R} \left( \frac{1}{T} \right) + \ln D_0

that can be fitted and used to extrapolate the diffusion coefficient at room
temperature, which is usually difficult to obtain directly, or to get the
activation energy.
"""

# ============================================================================
# CONSTANTS
# ============================================================================

__author__ = "Francisco Fernandez"
__email__ = "fernandezfrancisco2195@gmail.com"
__version__ = "0.1.0"


# ============================================================================
# IMPORTS
# ============================================================================

from .arrhenius import ArrheniusDiffusion  # noqa
from .diffusion import get_diffusion_coefficient  # noqa
