#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pytest

import sierras.arrhenius

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize(
    ("temps", "dcoeffs", "dcoeffserr", "ref", "decimal"),
    [
        (  # roughly equivalent to Fuller 1953 silicon data.
            np.array([1250, 1153.36, 1063.13, 970.65, 861.04, 769.34]),
            np.array(
                [
                    7.72104e-06,
                    4.386714e-06,
                    2.23884e-06,
                    5.58574e-07,
                    5.15115e-07,
                    7.58213e-08,
                ]
            ),
            np.array(
                [
                    1.42028e-06,
                    9.239103e-07,
                    6.98605e-07,
                    1.93034e-07,
                    1.18240e-07,
                    2.85640e-09,
                ]
            ),
            (-8513.868, -5.089957),
            (3, 3),
        ),
        (  # roughly equivalent to de Souza LJ 2006 data.
            np.array(
                [
                    1217.694563,
                    934.963910,
                    863.118100,
                    792.707095,
                    734.074259,
                    659.996304,
                    597.864428,
                    537.747162,
                    474.885671,
                    414.531828,
                    356.332201,
                ]
            ),
            np.array(
                [
                    0.031304,
                    0.020066,
                    0.017822,
                    0.014099,
                    0.011692,
                    0.008660,
                    0.007094,
                    0.004650,
                    0.003090,
                    0.001521,
                    0.000681,
                ]
            ),
            None,
            (-1919.8839, -1.8267787),
            (4, 7),
        ),
    ],
)
def test_arrhenius_diffusion_fit(temps, dcoeffs, dcoeffserr, ref, decimal):
    """Test the ArrheniusDiffusion class, fit method."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    slope, intercept = arrhenius.fit()

    np.testing.assert_almost_equal(slope.magnitude, ref[0], decimal[0])
    np.testing.assert_almost_equal(intercept.magnitude, ref[1], decimal[1])

    assert str(slope.units) == "kelvin"
    assert str(intercept.units) == "dimensionless"


@pytest.mark.parametrize(
    ("temps", "dcoeffs", "dcoeffserr", "ref"),
    [
        (  # roughly equivalent to Fuller 1953 silicon data.
            np.array([1250, 1153.36, 1063.13, 970.65, 861.04, 769.34]),
            np.array(
                [
                    7.72104e-06,
                    4.386714e-06,
                    2.23884e-06,
                    5.58574e-07,
                    5.15115e-07,
                    7.58213e-08,
                ]
            ),
            np.array(
                [
                    1.42028e-06,
                    9.239103e-07,
                    6.98605e-07,
                    1.93034e-07,
                    1.18240e-07,
                    2.85640e-09,
                ]
            ),
            (2.9132e-15, 2.9348e-15),
        ),
        (  # roughly equivalent to de Souza LJ 2006 data.
            np.array(
                [
                    1217.694563,
                    934.963910,
                    863.118100,
                    792.707095,
                    734.074259,
                    659.996304,
                    597.864428,
                    537.747162,
                    474.885671,
                    414.531828,
                    356.332201,
                ]
            ),
            np.array(
                [
                    0.031304,
                    0.020066,
                    0.017822,
                    0.014099,
                    0.011692,
                    0.008660,
                    0.007094,
                    0.004650,
                    0.003090,
                    0.001521,
                    0.000681,
                ]
            ),
            None,
            (0.0002674998, None),
        ),
    ],
)
def test_arrhenius_diffusion_extrapolate(temps, dcoeffs, dcoeffserr, ref):
    """Test the ArrheniusDiffusion class, extrapolate method."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    arrhenius.fit()
    damb, damberr = arrhenius.extrapolate()

    np.testing.assert_almost_equal(damb.magnitude, ref[0])
    if ref[1] is None:
        assert damberr is ref[1]
    else:
        np.testing.assert_almost_equal(damberr.magnitude, ref[1])

    assert str(damb.units) == "centimeter ** 2 / second"


@pytest.mark.parametrize(
    ("temps", "dcoeffs", "dcoeffserr", "ref"),
    [
        (  # roughly equivalent to Fuller 1953 silicon data.
            np.array([1250, 1153.36, 1063.13, 970.65, 861.04, 769.34]),
            np.array(
                [
                    7.72104e-06,
                    4.386714e-06,
                    2.23884e-06,
                    5.58574e-07,
                    5.15115e-07,
                    7.58213e-08,
                ]
            ),
            np.array(
                [
                    1.42028e-06,
                    9.239103e-07,
                    6.98605e-07,
                    1.93034e-07,
                    1.18240e-07,
                    2.85640e-09,
                ]
            ),
            0.7336684,
        ),
        (  # roughly equivalent to de Souza LJ 2006 data.
            np.array(
                [
                    1217.694563,
                    934.963910,
                    863.118100,
                    792.707095,
                    734.074259,
                    659.996304,
                    597.864428,
                    537.747162,
                    474.885671,
                    414.531828,
                    356.332201,
                ]
            ),
            np.array(
                [
                    0.031304,
                    0.020066,
                    0.017822,
                    0.014099,
                    0.011692,
                    0.008660,
                    0.007094,
                    0.004650,
                    0.003090,
                    0.001521,
                    0.000681,
                ]
            ),
            None,
            0.16544279,
        ),
    ],
)
def test_arrhenius_diffusion_activation_energy(
    temps, dcoeffs, dcoeffserr, ref
):
    """Test the ArrheniusDiffusion class, activation energy method."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    arrhenius.fit()
    acteng = arrhenius.activation_energy()

    np.testing.assert_almost_equal(acteng.magnitude, ref)
    assert str(acteng.units) == "electron_volt"


@pytest.mark.parametrize(
    ("temps", "dcoeffs", "dcoeffserr"),
    [
        (  # roughly equivalent to Fuller 1953 silicon data.
            np.array([1250, 1153.36, 1063.13, 970.65, 861.04, 769.34]),
            np.array(
                [
                    7.72104e-06,
                    4.386714e-06,
                    2.23884e-06,
                    5.58574e-07,
                    5.15115e-07,
                    7.58213e-08,
                ]
            ),
            np.array(
                [
                    1.42028e-06,
                    9.239103e-07,
                    6.98605e-07,
                    1.93034e-07,
                    1.18240e-07,
                    2.85640e-09,
                ]
            ),
        ),
        (  # roughly equivalent to de Souza LJ 2006 data.
            np.array(
                [
                    1217.694563,
                    934.963910,
                    863.118100,
                    792.707095,
                    734.074259,
                    659.996304,
                    597.864428,
                    537.747162,
                    474.885671,
                    414.531828,
                    356.332201,
                ]
            ),
            np.array(
                [
                    0.031304,
                    0.020066,
                    0.017822,
                    0.014099,
                    0.011692,
                    0.008660,
                    0.007094,
                    0.004650,
                    0.003090,
                    0.001521,
                    0.000681,
                ]
            ),
            None,
        ),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.005)
def test_arrhenius_diffusion_plot(
    fig_test, fig_ref, temps, dcoeffs, dcoeffserr
):
    """Test the ArrheniusDiffusion class, plots."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    slope, intercept = arrhenius.fit()

    # test
    test_ax = fig_test.subplots()
    arrhenius.plot(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()
    exp_ax.errorbar(
        1 / temps,
        np.log(dcoeffs),
        yerr=dcoeffserr / dcoeffs if dcoeffserr is not None else None,
        marker="o",
        ls="",
        label="diffusion",
    )
    exp_ax.plot(
        1 / temps, intercept.magnitude + slope.magnitude / temps, label="fit"
    )


@pytest.mark.parametrize(
    ("temps", "dcoeffs", "dcoeffserr", "reffname"),
    [
        (  # roughly equivalent to Fuller 1953 silicon data.
            np.array([1250, 1153.36, 1063.13, 970.65, 861.04, 769.34]),
            np.array(
                [
                    7.72104e-06,
                    4.386714e-06,
                    2.23884e-06,
                    5.58574e-07,
                    5.15115e-07,
                    7.58213e-08,
                ]
            ),
            np.array(
                [
                    1.42028e-06,
                    9.239103e-07,
                    6.98605e-07,
                    1.93034e-07,
                    1.18240e-07,
                    2.85640e-09,
                ]
            ),
            "fuller53-Si.csv",
        ),
        (  # roughly equivalent to de Souza LJ 2006 data.
            np.array(
                [
                    1217.694563,
                    934.963910,
                    863.118100,
                    792.707095,
                    734.074259,
                    659.996304,
                    597.864428,
                    537.747162,
                    474.885671,
                    414.531828,
                    356.332201,
                ]
            ),
            np.array(
                [
                    0.031304,
                    0.020066,
                    0.017822,
                    0.014099,
                    0.011692,
                    0.008660,
                    0.007094,
                    0.004650,
                    0.003090,
                    0.001521,
                    0.000681,
                ]
            ),
            None,
            "desouza06-LJ.csv",
        ),
    ],
)
def test_arrhenius_diffusion_to_csv(temps, dcoeffs, dcoeffserr, reffname):
    """Test the ArrheniusDiffusion class, save to csv."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    arrhenius.fit()

    arrhenius_test_data = str(TEST_DATA_PATH / "arrhenius.csv")
    arrhenius.to_csv(path_or_buf=arrhenius_test_data)

    with open(arrhenius_test_data, "r") as f:
        writed = f.read()
    os.remove(arrhenius_test_data)

    with open(TEST_DATA_PATH / reffname, "r") as f:
        expected = f.read()

    assert writed == expected
