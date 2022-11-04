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

import pandas as pd

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
            (-9196.819, -4.394486),
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
            (3, 5),
        ),
        (  # roughly equivalent to de Wei-Zhong LJ 2008 Kubo-Green data.
            np.array(
                [
                    0.7000154,
                    0.80037778,
                    0.90050338,
                    1.00071451,
                    1.10114817,
                    1.20103096,
                ]
            )
            * 1_000,
            np.array(
                [
                    0.03800871,
                    0.04596339,
                    0.05495668,
                    0.0619257,
                    0.07191,
                    0.08090012,
                ]
            ),
            None,
            (-1258.7578, -1.49366295),
            (3, 5),
        ),
    ],
)
def test_arrhenius_diffusion_fit(temps, dcoeffs, dcoeffserr, ref, decimal):
    """Test the ArrheniusDiffusion class, fit method."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    model = arrhenius.fit()

    np.testing.assert_almost_equal(model.slope_.magnitude, ref[0], decimal[0])
    np.testing.assert_almost_equal(
        model.intercept_.magnitude, ref[1], decimal[1]
    )

    assert str(model.slope_.units) == "kelvin"
    assert str(model.intercept_.units) == "dimensionless"


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
        (  # roughly equivalent to de Wei-Zhong LJ 2008 Kubo-Green data.
            np.array(
                [
                    0.7000154,
                    0.80037778,
                    0.90050338,
                    1.00071451,
                    1.10114817,
                    1.20103096,
                ]
            )
            * 1_000,
            np.array(
                [
                    0.03800871,
                    0.04596339,
                    0.05495668,
                    0.0619257,
                    0.07191,
                    0.08090012,
                ]
            ),
            None,
            (0.0033812, None),
        ),
    ],
)
def test_arrhenius_diffusion_extrapolate(temps, dcoeffs, dcoeffserr, ref):
    """Test the ArrheniusDiffusion class, extrapolate method."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    arrhenius.fit()
    damb = arrhenius.extrapolate()

    if ref[1] is None:
        np.testing.assert_almost_equal(damb.magnitude, ref[0])
    else:
        np.testing.assert_almost_equal(damb.value.magnitude, ref[0])
        np.testing.assert_almost_equal(damb.error.magnitude, ref[1])

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
            (0.79252057, 0.0243509),
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
            (0.16544279, None),
        ),
        (  # roughly equivalent to de Wei-Zhong LJ 2008 Kubo-Green data.
            np.array(
                [
                    0.7000154,
                    0.80037778,
                    0.90050338,
                    1.00071451,
                    1.10114817,
                    1.20103096,
                ]
            )
            * 1_000,
            np.array(
                [
                    0.03800871,
                    0.04596339,
                    0.05495668,
                    0.0619257,
                    0.07191,
                    0.08090012,
                ]
            ),
            None,
            (0.1084714, None),
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

    if ref[1] is None:
        np.testing.assert_almost_equal(acteng.magnitude, ref[0])
    else:
        np.testing.assert_almost_equal(acteng.value.magnitude, ref[0])
        np.testing.assert_almost_equal(acteng.error.magnitude, ref[1])

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
        (  # roughly equivalent to de Wei-Zhong LJ 2008 Kubo-Green data.
            np.array(
                [
                    0.7000154,
                    0.80037778,
                    0.90050338,
                    1.00071451,
                    1.10114817,
                    1.20103096,
                ]
            )
            * 1_000,
            np.array(
                [
                    0.03800871,
                    0.04596339,
                    0.05495668,
                    0.0619257,
                    0.07191,
                    0.08090012,
                ]
            ),
            None,
        ),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.01)
def test_arrhenius_diffusion_plot(
    fig_test, fig_ref, temps, dcoeffs, dcoeffserr
):
    """Test the ArrheniusDiffusion class, plots."""
    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    model = arrhenius.fit()
    slope, intercept = model.slope_, model.intercept_

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
        (  # roughly equivalent to de Wei-Zhong LJ 2008 Kubo-Green data.
            np.array(
                [
                    0.7000154,
                    0.80037778,
                    0.90050338,
                    1.00071451,
                    1.10114817,
                    1.20103096,
                ]
            )
            * 1_000,
            np.array(
                [
                    0.03800871,
                    0.04596339,
                    0.05495668,
                    0.0619257,
                    0.07191,
                    0.08090012,
                ]
            ),
            None,
            "wei-zhong08-KG.csv",
        ),
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
    ],
)
def test_arrhenius_diffusion_to_csv(temps, dcoeffs, dcoeffserr, reffname):
    """Test the ArrheniusDiffusion class, save to csv."""
    df_ref = pd.read_csv(str(TEST_DATA_PATH / reffname), dtype=np.float32)

    arrhenius = sierras.arrhenius.ArrheniusDiffusion(
        temps, dcoeffs, differr=dcoeffserr
    )
    arrhenius.fit()
    df = arrhenius.to_dataframe()

    pd.testing.assert_frame_equal(df, df_ref)
