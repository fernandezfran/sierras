#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import sierras.arrhenius

# ============================================================================
# TESTS
# ============================================================================


def test_arrhenius_diffusion_fit():
    """Test the ArrheniusDiffusion class."""
    # artificial data
    temps = np.arange(900, 1300, 100)
    dcoeffs = np.array([1e-10, 1e-9, 1e-8, 1e-7])

    arrhenius = sierras.arrhenius.ArrheniusDiffusion(temps, dcoeffs)
    slope, intercept = arrhenius.fit()

    np.testing.assert_almost_equal(slope, -24731.535, 2)
    np.testing.assert_almost_equal(intercept, 4.253992)


def test_arrhenius_diffusion_extrapolate():
    """Test the ArrheniusDiffusion class."""
    # artificial data
    temps = np.arange(900, 1300, 100)
    dcoeffs = np.array([4e-8, 6e-8, 8e-8, 1e-7])

    arrhenius = sierras.arrhenius.ArrheniusDiffusion(temps, dcoeffs)
    arrhenius.fit()
    damb, damberr = arrhenius.extrapolate()

    np.testing.assert_almost_equal(damb, 2.6610053e-11)
    assert damberr is None


@check_figures_equal(extensions=["png"])
def test_arrhenius_diffusion_plot(fig_test, fig_ref):
    """Test the ArrheniusDiffusion class."""
    # artificial data
    temps = np.arange(900, 1300, 100)
    dcoeffs = np.array([4e-8, 6e-8, 8e-8, 1e-7])

    arrhenius = sierras.arrhenius.ArrheniusDiffusion(temps, dcoeffs)
    slope, intercept = arrhenius.fit()

    # test
    test_ax = fig_test.subplots()
    arrhenius.plot(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()
    exp_ax.errorbar(
        1 / temps, np.log(dcoeffs), marker="o", ls="", label="diffusion"
    )
    exp_ax.plot(1 / temps, intercept + slope / temps, label="fit")
