#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Function to fit the mean square displacement."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import sklearn.linear_model

# ============================================================================
# FUNCTIONS
# ============================================================================


def get_diffusion_coefficient(
    time, msd, start=0, stop=None, ndim=3, sample_weight=None, lr_kws=None
):
    """To obtain the trace diffusion coefficient from mean square displacement.

    Parameters
    ----------
    time : array-like
        time data.

    msd : array-like
        mean square displacement data.

    start : int, default=0
        the first value to be considered in the fit.

    stop : int, default=None
        the last value to be considered in the fit.

    ndim : int, defualt=3
        the dimension of the system.

    sample_weight: array-like, default=None
        as described in the method fit of
        `sklearn.linear_model.LinearRegression`.

    lr_kws : dict
        additional keyword arguments that are passed and are documented in
        `sklearn.linear_model.LinearRegression`.

    Returns
    -------
    tuple
        a tuple where the first element is a float with the fitted diffusion
        coefficient and the second element is a object of
        `sklearn.linear_model.LinearRegression`, where different attributes
        can be obteined (slope, intercept, etc).
    """
    stop = len(time) if stop is None else stop

    lr_kws = {} if lr_kws is None else lr_kws

    time = np.asarray(time, dtype=np.float32)[start:stop].reshape(-1, 1)
    msd = np.asarray(msd, dtype=np.float32)[start:stop]

    reg = sklearn.linear_model.LinearRegression(**lr_kws).fit(
        time, msd, sample_weight
    )

    dcoef = reg.coef_[0] / (2 * ndim)

    return dcoef, reg
