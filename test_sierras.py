#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021-2023, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pytest

import sierras

# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize(("dataset"), ["fuller53", "desouza06", "weizhong08"])
class TestArrheniusRegressor:
    """Test the Arrhenius Regressor."""

    def test_fit(self, dataset, request):
        """Test the empirical Arrhenius equation fitting."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor()
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        np.testing.assert_almost_equal(
            areg.reg_.coef_[0], dataset["ref"]["slope"]
        )
        np.testing.assert_almost_equal(
            areg.reg_.intercept_, dataset["ref"]["intercept"]
        )

    def test_predict(self, dataset, request):
        """Test the prediction of the thermally-induced process."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor()
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        pred = areg.predict(dataset["X"])

        np.testing.assert_almost_equal(pred, dataset["ref"]["pred"])

    def test_to_dataframe(self, dataset, request):
        """Test the dataframe."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor()
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        pred = areg.predict(dataset["X"])
        df_ref = pd.DataFrame(
            {
                "temperatures": dataset["X"].ravel(),
                "process": dataset["y"],
                "process_pred": pred,
            }
        )
        if dataset["sample_weight"] is not None:
            df_ref["weights"] = dataset["sample_weight"]

        df = areg.to_dataframe(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        assert df.equals(df_ref)

    def test_activation_energy(self, dataset, request):
        """Test the activation energy."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor()
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        np.testing.assert_almost_equal(
            areg.activation_energy_, dataset["ref"]["activation_energy"]
        )

    def test_extrapolation_at_room_temperature(self, dataset, request):
        """Test the extrapolation at room temperature of the process."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor()
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        room_process = areg.predict(np.array([[300.00]]))

        np.testing.assert_almost_equal(
            room_process, dataset["ref"]["room_process"], decimal=16
        )

    @check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
    def test_plot(self, fig_test, fig_ref, dataset, request):
        """Test the Arrhenius Plotter."""
        dataset = request.getfixturevalue(dataset)

        X, y, sample_weight = (
            dataset["X"],
            dataset["y"],
            dataset["sample_weight"],
        )

        areg = sierras.ArrheniusRegressor()
        areg.fit(X, y, sample_weight=sample_weight)
        pred = areg.predict(dataset["X"])

        # test
        test_ax = fig_test.subplots()
        areg.plot(ax=test_ax)

        # expected
        ref_ax = fig_ref.subplots()
        ref_ax.errorbar(
            1 / X,
            np.log(y),
            yerr=sample_weight / y if sample_weight is not None else None,
            marker="o",
            ls="",
            label="data",
        )
        ref_ax.plot(1 / X, np.log(pred), label="fit")


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture()
def fuller53():
    return {
        "X": np.array(
            [1250, 1153.36, 1063.13, 970.65, 861.04, 769.34]
        ).reshape(-1, 1),
        "y": np.array(
            [
                7.72104e-06,
                4.386714e-06,
                2.23884e-06,
                5.58574e-07,
                5.15115e-07,
                7.58213e-08,
            ]
        ),
        "sample_weight": np.array(
            [
                1.42028e-06,
                9.239103e-07,
                6.98605e-07,
                1.93034e-07,
                1.18240e-07,
                2.85640e-09,
            ]
        ),
        "ref": {
            "slope": -8513.869191,
            "intercept": -5.0899556,
            "activation_energy": 0.7336685,
            "pred": np.array(
                [
                    6.7832721e-06,
                    3.8334229e-06,
                    2.0487914e-06,
                    9.5527874e-07,
                    3.1275489e-07,
                    9.6240639e-08,
                ]
            ),
            "room_process": np.array([2.913213627046838e-15]),
        },
    }


@pytest.fixture()
def desouza06():
    return {
        "X": np.array(
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
        ).reshape(-1, 1),
        "y": np.array(
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
        "sample_weight": None,
        "ref": {
            "slope": -1919.8841057,
            "intercept": -1.8267783,
            "activation_energy": 0.1654428,
            "pred": np.array(
                [
                    0.0332589,
                    0.0206465,
                    0.0174026,
                    0.0142826,
                    0.0117705,
                    0.0087762,
                    0.0064866,
                    0.0045301,
                    0.002824,
                    0.0015676,
                    0.0007357,
                ]
            ),
            "room_process": np.array([0.0002674997469623]),
        },
    }


@pytest.fixture()
def weizhong08():
    return {
        "X": (
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
            * 1_000
        ).reshape(-1, 1),
        "y": np.array(
            [
                0.03800871,
                0.04596339,
                0.05495668,
                0.0619257,
                0.07191,
                0.08090012,
            ]
        ),
        "sample_weight": None,
        "ref": {
            "slope": -1258.7577978,
            "intercept": -1.4936627,
            "activation_energy": 0.1084714,
            "pred": np.array(
                [
                    0.037185,
                    0.0465901,
                    0.0554929,
                    0.0638307,
                    0.0715904,
                    0.0787303,
                ]
            ),
            "room_process": np.array([0.0033812088051859]),
        },
    }
