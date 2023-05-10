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
class TestArrheniusPlotter:
    """Test the Arrhenius Plotter."""

    @check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
    def test_plot(self, fig_test, fig_ref, dataset, request):
        """Test the arrhenius-plot function."""
        dataset = request.getfixturevalue(dataset)

        X, y, sample_weight = (
            dataset["X"],
            dataset["y"],
            dataset["sample_weight"],
        )

        areg = sierras.ArrheniusRegressor(BOLTZMANN)
        areg.fit(X, y, sample_weight=sample_weight)
        pred = areg.predict(dataset["X"])

        # test
        test_ax = fig_test.subplots()
        areg.plot.arrhenius(X, y, ax=test_ax)

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
