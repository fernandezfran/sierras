#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import pytest

import sierras.get_diffusion_coefficient

# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize(
    ("start", "stop", "reference"),
    [
        (0, None, (0.64941016, (3.896461, 0.564003))),
        (50, 150, (0.62004729, (3.7202837, 1.6425037))),
        (0, 150, (0.65676272, (3.9405763, 0.4432726))),
        (50, None, (0.63124760, (3.7874856, 1.333454))),
        (0, 50, (0.70261868, (4.215712, -0.1003175))),
    ],
)
def test_get_diffusion_coefficient_from_msd(start, stop, reference):
    """Test the linear regression to get the trace diffusion coefficient.

    This test uses data of a Lennard-Jones fluid in its liquid phase.
    """
    time_data = 0.05 * np.arange(0, 200)
    msd_data = np.array(
        [
            0.0,
            0.18579898,
            0.35747746,
            0.5426987,
            0.73278261,
            0.9338407,
            1.13437932,
            1.36205518,
            1.56029879,
            1.73697416,
            1.96884115,
            2.18462587,
            2.37341302,
            2.57218206,
            2.82805379,
            3.12232283,
            3.29623498,
            3.45791592,
            3.65135412,
            3.87218877,
            4.09262591,
            4.40193404,
            4.63385581,
            4.80765401,
            5.05222344,
            5.22308566,
            5.40990772,
            5.67813274,
            5.81758522,
            6.0462092,
            6.26295127,
            6.41525821,
            6.48066494,
            6.78124055,
            6.95534262,
            7.22674676,
            7.36488895,
            7.52730555,
            7.89060876,
            8.13552497,
            8.33028478,
            8.56743357,
            8.8541472,
            9.01204869,
            9.2139022,
            9.36187584,
            9.43919832,
            9.80264101,
            10.15579164,
            10.38198812,
            10.69345633,
            10.90015012,
            11.11918441,
            11.32155051,
            11.56063097,
            11.769107,
            11.99279092,
            12.09625331,
            12.3371963,
            12.55848919,
            12.5981481,
            12.73328404,
            12.95890588,
            13.17773588,
            13.37676504,
            13.61786791,
            13.88813064,
            14.15893945,
            14.33814374,
            14.41392676,
            14.53318692,
            14.72803508,
            14.74063587,
            14.82099437,
            14.99495963,
            15.16357773,
            15.45726625,
            15.71460651,
            16.1032025,
            16.36619087,
            16.46001708,
            16.54509427,
            16.62046373,
            16.92323184,
            17.15111028,
            17.32838805,
            17.44211325,
            17.67580442,
            18.01044074,
            17.99441144,
            18.2475785,
            18.56229178,
            18.8592032,
            19.1061905,
            19.03559045,
            19.1955458,
            19.29038906,
            19.41953264,
            19.56530818,
            19.98805892,
            20.49746008,
            20.83550011,
            20.98118283,
            21.2237299,
            21.5608569,
            21.72466822,
            21.96838882,
            22.38860688,
            22.60096692,
            22.86585386,
            23.07524908,
            23.25634218,
            23.33714576,
            23.34287702,
            23.70998703,
            23.81080757,
            23.90971071,
            24.19646691,
            24.43037671,
            24.32978941,
            24.35533812,
            24.57236711,
            24.7618253,
            25.11521481,
            25.28103188,
            25.30450023,
            25.48309271,
            25.55583137,
            25.55879713,
            25.68217475,
            25.73247962,
            26.00844048,
            26.06215095,
            26.26288579,
            26.40935604,
            26.56342122,
            26.65723871,
            26.6508749,
            26.79690477,
            26.81037309,
            26.95303199,
            27.15802999,
            27.46386169,
            27.52098757,
            27.66517052,
            27.97374278,
            28.08132325,
            28.17014195,
            28.34778873,
            28.43299574,
            28.65420689,
            28.88235334,
            29.37324761,
            29.72699006,
            29.86516599,
            30.09353786,
            30.49408872,
            31.00501446,
            31.24532821,
            31.57681322,
            31.99763296,
            32.1887299,
            32.62616242,
            32.57142794,
            32.84459969,
            32.90820018,
            33.1109161,
            33.47233807,
            33.74549947,
            33.9370431,
            34.11598648,
            34.41971601,
            34.41428703,
            34.24803916,
            34.52027277,
            34.44971153,
            34.52997185,
            34.79932404,
            34.89158712,
            35.24137191,
            35.2698819,
            35.59213711,
            35.74307491,
            35.77802544,
            36.23085136,
            36.31178006,
            36.48317429,
            36.65046282,
            36.77030673,
            37.18797419,
            37.51268828,
            37.82048554,
            37.87564568,
            38.0113201,
            38.13147069,
            38.33394108,
            38.62516787,
            38.89001971,
            38.96842628,
            39.35482202,
        ]
    )

    dcoeff, reg = sierras.get_diffusion_coefficient.from_msd(
        time_data, msd_data, start=start, stop=stop
    )

    np.testing.assert_almost_equal(dcoeff, reference[0])
    np.testing.assert_almost_equal(reg.coef_[0], reference[1][0])
    np.testing.assert_almost_equal(reg.intercept_, reference[1][1])
