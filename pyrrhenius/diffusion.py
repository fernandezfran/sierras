#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt

import sklearn.linear_model


def get_diffusion_coefficient(
        t, msd, start=0, stop=None, fit_intercept=False, dim=3
):
    """To obtain the trace diffusion coefficient from MSD."""
    stop = len(t) if stop is None else stop

    t = np.asarray(t, dtype=np.float32)
    msd = np.asarray(msd, dtype=np.float32)

    X = t[start:stop].reshape(-1, 1)
    y = msd[start:stop]

    reg = sklearn.linear_model.LinearRegression(
        fit_intercept=fit_intercept
    ).fit(X, y)

    slope = reg.coef_[0]
    intercept = reg.intercept_

    return slope / (2 * dim), (slope, intercept)
