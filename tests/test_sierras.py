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

import pandas as pd

import pytest

import sierras

# ============================================================================
# CONSTANTS
# ============================================================================

BOLTZMANN = 8.617333262e-5

# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize(("dataset"), ["fuller53", "desouza06", "weizhong08"])
class TestArrheniusRegressor:
    """Test the Arrhenius Regressor."""

    def test_fit(self, dataset, request):
        """Test the empirical Arrhenius equation fitting."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor(BOLTZMANN)
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

        areg = sierras.ArrheniusRegressor(BOLTZMANN)
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        pred = areg.predict(dataset["X"])

        np.testing.assert_almost_equal(pred, dataset["ref"]["pred"])

    def test_to_dataframe(self, dataset, request):
        """Test the dataframe."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor(BOLTZMANN)
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

        areg = sierras.ArrheniusRegressor(BOLTZMANN)
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        np.testing.assert_almost_equal(
            areg.activation_energy_, dataset["ref"]["activation_energy"]
        )

    def test_extrapolation_at_room_temperature(self, dataset, request):
        """Test the extrapolation at room temperature of the process."""
        dataset = request.getfixturevalue(dataset)

        areg = sierras.ArrheniusRegressor(BOLTZMANN)
        areg.fit(
            dataset["X"], dataset["y"], sample_weight=dataset["sample_weight"]
        )

        room_process = areg.predict(np.array([[300.00]]))

        np.testing.assert_almost_equal(
            room_process, dataset["ref"]["room_process"], decimal=16
        )
