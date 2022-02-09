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

import pint

import sklearn.linear_model

# ============================================================================
# CONSTANTS
# ============================================================================

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

pint.set_application_registry(ureg)  # i am not sure of this

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

    sysunits : dict, optional.
        the system of units, default value is:
        {"temperature": "kelvin", "distance": "centimeters", "time": "seconds",
        "energy": "eV"}
    """

    def __init__(
        self,
        temperatures,
        diffusion_coefficients,
        differr=None,
        temperr=None,
        sysunits=None,
    ):
        # first of all, I define the units
        self.sysunits = {} if sysunits is None else sysunits
        for key, value in zip(
            ("temperature", "distance", "time", "energy"),
            ("kelvin", "centimeter", "second", "eV"),
        ):
            self.sysunits.setdefault(key, value)

        # force the temperatures to be in kelvin
        self.temperatures = (
            Q_(
                np.asarray(temperatures, dtype=np.float32),
                ureg(self.sysunits["temperature"]),
            )
        ).to("kelvin")
        self.temperr = (
            (
                Q_(
                    np.asarray(temperr, dtype=np.float32),
                    ureg(self.sysunits["temperature"]),
                )
            ).to("kelvin")
            if temperr is not None
            else None
        )

        self.diffusion_coefficients = (
            np.array(diffusion_coefficients, dtype=np.float32)
            * ureg(self.sysunits["distance"]) ** 2
            / ureg(self.sysunits["time"])
        )
        self.differr = (
            np.asarray(differr, dtype=np.float32)
            * ureg(self.sysunits["distance"]) ** 2
            / ureg(self.sysunits["time"])
            if differr is not None
            else None
        )

        # change to the variables ln(D) y 1/T
        self.tempinv_ = 1 / self.temperatures.magnitude
        self.diff_ = np.log(self.diffusion_coefficients.magnitude)

        # error propagation by partial derivatives:
        self.tempinv_err_ = (
            self.temperr.magnitude / (self.temperatures.magnitude ** 2)  # noqa
            if self.temperr is not None
            else None
        )
        self.diff_err_ = (
            self.differr.magnitude / self.diffusion_coefficients.magnitude
            if self.differr is not None
            else None
        )

    @property
    def _delta_error(self):
        """Slope and intercept error common denominator."""
        return (
            np.sum(1 / self.diff_err_ ** 2)
            * np.sum((self.tempinv_ / self.diff_err_) ** 2)
            - np.sum(self.tempinv_ / self.diff_err_ ** 2) ** 2
        )

    @property
    def _slope_error(self):
        """Error of the slope."""
        return np.sqrt(np.sum(1 / self.diff_err_ ** 2) / self._delta_error)

    @property
    def _intercept_error(self):
        """Error of the intercept."""
        return np.sqrt(
            np.sum((self.tempinv_ / self.diff_err_) ** 2) / self._delta_error
        )

    def _diff_error_propagation(self, dtemp):
        """Error of the diffusion coefficient extrapolation."""
        return self.dcoeff_ * np.sqrt(
            (self._slope_error / dtemp.to("kelvin").magnitude) ** 2
            + self._intercept_error ** 2
        )

    def fit(self, **kwargs):
        """Fit linear model.

        Parameters
        ----------
        **kwargs
            additional keyword arguments that are passed and are documented in
            `sklearn.linear_model.LinearRegression`.

        Attributes
        ----------
        slope_ : `pint.UnitRegistry.Quantity`
            the fitted slope in Kelvin units.

        intercept_ : `pint.UnitRegistry.Quantity`
            the dimensionless intercept.
        """
        reg = sklearn.linear_model.LinearRegression(**kwargs).fit(
            self.tempinv_.reshape(-1, 1),
            self.diff_,
            sample_weight=1 / self.diff_err_
            if self.diff_err_ is not None
            else None,
        )

        self.slope_ = Q_(reg.coef_[0], ureg(self.sysunits["temperature"])).to(
            "kelvin"
        )
        self.intercept_ = reg.intercept_ * ureg("dimensionless")

        return self

    def extrapolate(self, dtemp=300.0):
        """Extrapolates the diffusion coefficient at a desired temperature.

        Parameters
        ----------
        dtemp : float, default=300.0
            the desired temperature for the extrapolation.

        Returns
        -------
        `pint.UnitRegistry.Quantity`
            a `pint.UnitRegistry.Quantity` object with the extrapolated
            diffusion coefficient at the desired temperature in the specified
            units of the system (distance ** 2 / time) and the respective
            error if this was possible to calculate.
        """
        dtemp = Q_(dtemp, ureg(self.sysunits["temperature"]))
        self.dcoeff_ = np.exp(
            self.slope_.magnitude / dtemp.to("kelvin").magnitude
            + self.intercept_.magnitude
        )
        self.dcoefferr_ = (
            self._diff_error_propagation(dtemp)
            if self.diff_err_ is not None
            else None
        )

        self.dcoeff_ = (
            self.dcoeff_
            * ureg(self.sysunits["distance"]) ** 2
            / ureg(self.sysunits["time"])
        )
        self.dcoefferr_ = (
            self.dcoefferr_
            * ureg(self.sysunits["distance"]) ** 2
            / ureg(self.sysunits["time"])
            if self.dcoefferr_ is not None
            else None
        )

        return (
            self.dcoeff_
            if self.dcoefferr_ is None
            else self.dcoeff_.plus_minus(self.dcoefferr_)
        )

    def activation_energy(self):
        """Get the activation energy from the fit slope.

        Returns
        -------
        float
            a `pint.UnitRegistry.Quantity` object with the activation energy
            of the diffusive process in system energy units and the respective
            error if this was possible to calculate.
        """
        r_ideal_gas = Q_("boltzmann_constant").to(
            ureg(self.sysunits["energy"]) / ureg("kelvin")
        )
        act_eng = -self.slope_ * r_ideal_gas

        return (
            act_eng
            if self.diff_err_ is None
            else act_eng.plus_minus(self._slope_error * r_ideal_gas.magnitude)
        )

    def predict(self, temperatures):
        """Predict using the linear model.

        Parameters
        ----------
        temperatures : array-like
            the temperatures at which you want to predict the diffusion
            coefficient.
        """
        return self.intercept_.magnitude + self.slope_.magnitude * temperatures

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
        if self.slope_ or self.intercept_:
            ax.plot(self.tempinv_, self.predict(self.tempinv_), **plot_kws)

        return ax

    def to_dataframe(self):
        """Convert the results to a pandas.DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with all the data.
        """
        dict_ = {
            "temperatures-inv": self.tempinv_,
            "values-diffusion-log": self.diff_,
            "extrapolated-diffusion-log": self.predict(self.tempinv_),
        }

        if self.diff_err_ is not None:
            dict_["values-err-diffusion-log"] = self.diff_err_

        if self.tempinv_err_ is not None:
            dict_["temperatures-inv-err"] = self.tempinv_err_

        return pd.DataFrame(dict_)
