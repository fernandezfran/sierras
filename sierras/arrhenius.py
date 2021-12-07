#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Arrhenius fit, plot and extrapolation."""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn.linear_model

# ============================================================================
# CONSTANTS
# ============================================================================

R_eV = 8.3145 * 1.03636e-5

# ============================================================================
# CLASSES
# ============================================================================


class ArrheniusDiffusion:
    """Arrhenius Diffusion class.

    Parameters
    ----------
    temperatures, diffusion coefficient : array-like
        temperatures and corresponding diffusion coefficients data.

    temperr, differr : array-like, optional
        the error of each data point.
    """

    def __init__(
        self, temperatures, diffusion_coefficients, differr=None, temperr=None
    ):
        self.temperatures = np.asarray(temperatures, dtype=np.float32)
        self.diffusion_coefficients = np.array(
            diffusion_coefficients, dtype=np.float32
        )
        self.differr = differr
        self.temperr = temperr

        # change to the variables ln(D) y 1/T
        self.tempinv_ = 1 / self.temperatures
        self.diff_ = np.log(self.diffusion_coefficients)

        # error propagation by partial derivatives:
        self.tempinv_err_ = (
            np.asarray(self.temperr, dtype=np.float32)
            / (self.temperatures ** 2)
            if self.temperr is not None
            else None
        )
        self.diff_err_ = (
            np.asarray(self.differr, dtype=np.float32)
            / self.diffusion_coefficients
            if self.differr is not None
            else None
        )

    def _error_propagation(self, dtemp):
        """Error of the extrapolation."""
        delta = (
            np.sum(1 / self.diff_err_ ** 2)
            * np.sum((self.tempinv_ / self.diff_err_) ** 2)
            - np.sum(self.tempinv_ / self.diff_err_ ** 2) ** 2
        )
        errslope = np.sqrt(np.sum(1 / self.diff_err_ ** 2) / delta)
        errintercept = np.sqrt(
            np.sum((self.tempinv_ / self.diff_err_) ** 2) / delta
        )

        return self.dcoeff_ * np.sqrt(
            (errslope / dtemp) ** 2 + errintercept ** 2
        )

    def fit(self, **kwargs):
        """Fit linear model.

        Parameters
        ----------
        **kwargs
            additional keyword arguments that are passed and are documented in
            `sklearn.linear_model.LinearRegression`.

        Returns
        -------
        tuple
             a tuple with the fitted slope and intercept.
        """
        reg = sklearn.linear_model.LinearRegression(**kwargs).fit(
            self.tempinv_.reshape(-1, 1),
            self.diff_,
            sample_weight=self.diff_err_,
        )

        self.slope_ = reg.coef_[0]
        self.intercept_ = reg.intercept_

        return self.slope_, self.intercept_

    def extrapolate(self, dtemp=300.0):
        """Extrapolates the diffusion coefficient at a desired temperature.

        Parameters
        ----------
        dtemp : float, default=300.0
            the desired temperature for the extrapolation.

        Returns
        -------
        tuple
            a tuple with the extrapolated diffusion coefficient at the desired
            temperature as a first element and the respective error in the
            second, if this was not possible to calculate, then is None.
        """
        self.dcoeff_ = np.exp(self.slope_ / dtemp + self.intercept_)
        self.dcoefferr_ = (
            self._error_propagation(dtemp)
            if self.diff_err_ is not None
            else None
        )

        return self.dcoeff_, self.dcoefferr_

    def activation_energy(self):
        """Get the activation energy from the fit slope.

        Returns
        -------
        float
            the activation energy of the diffusive process.
        """
        return self.slope_ * R_eV

    def predict(self, temperatures):
        """Predict using the linear model.

        Parameters
        ----------
        temperatures : array-like
            the temperatures at which you want to predict the diffusion
            coefficient.
        """
        return self.intercept_ + self.slope_ * temperatures

    def plot(self, ax=None, errorbar_kws=None, plot_kws=None):
        """Arrhenius plot.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis, default=None
            the current axes.

        errorbar_kws : dict, default=None
            additional keyword arguments that are passed and are documented in
            `matplotlib.pyplot.errorbar`.

        plot_kws : dict, default=None
            additional keyword arguments that are passed and are documented in
            `matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.pyplot.Axis
            The current axes.
        """
        ax = plt.gca() if ax is None else ax

        errorbar_kws = {} if errorbar_kws is None else errorbar_kws
        plot_kws = {} if plot_kws is None else plot_kws

        for key, value in zip(
            ["marker", "ls", "label"], ["o", "", "diffusion"]
        ):
            errorbar_kws.setdefault(key, value)

        plot_kws.setdefault("label", "fit")

        ax.errorbar(
            self.tempinv_,
            self.diff_,
            yerr=self.diff_err_,
            xerr=self.tempinv_err_,
            **errorbar_kws,
        )
        ax.plot(self.tempinv_, self.predict(self.tempinv_), **plot_kws)

        return ax

    def to_csv(self, **kwargs):
        """Write the results to a comma-separated values (csv) file.

        Parameters
        ----------
        **kwargs
            additional keyword arguments that are passed and are documented in
            `pandas.DataFrame.to_csv`.
        """
        pd.DataFrame(
            {
                "temperatures-inv": self.tempinv_,
                "extrapolated-diffusion-log": self.predict(self.tempinv_),
            }
        ).to_csv(**kwargs)
