#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of sierras (https://github.com/fernandezfran/sierras/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/sierras/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

r"""A tool for empirical Arrhenius equation fitting.

In physical chemistry, the Arrhenius equation is an empirical formula for
temperature dependence of a process given by the following equation

.. math::
    k = k_0 e^{-E / k_B T}

where

- :math:`k` is the thermally-induced process (diffusion coefficient,
  frequency of collision, crystal vacancies, among others),

- :math:`k_0` is the pre-exponential factor,

- :math:`E` is the activation energy of the process,

- :math:`k_B` is the Boltzmann constant, it could also be the universal gas
  constant :math:`R`,

- :math:`T` is the Temperature in Kelvin,

The exponential factor in this equation gives the probability that the process
occurs and it denotes the fraction of atoms with energy greater than or equal
than :math:`E`.

Taking the natural logarithm of this equation yields to a linear relationship

.. math::
    \ln k = \ln k_0 - \frac{E}{k_B} \left( \frac{1}{T} \right)

that can be fitted and used to extrapolate the process to room temperature,
which is usually difficult to obtain directly, or to get the activation
energy from the slope.

For more details read: https://en.wikipedia.org/wiki/Arrhenius_equation and/or
https://en.wikipedia.org/wiki/Arrhenius_plot
"""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib_metadata

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn.linear_model
from sklearn.base import BaseEstimator, RegressorMixin

# =============================================================================
# CONSTANTS
# =============================================================================

__all__ = ["ArrheniusRegressor", "ArrheniusPlotter"]


NAME = "sierras"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata

# =============================================================================
# CLASSES
# =============================================================================


class ArrheniusRegressor(BaseEstimator, RegressorMixin):
    r"""Arrhenius equation fitting.

    Parameters
    ----------
    constant : float
        Either the universal gas constant, :math:`R`, or the Boltzmann
        constant, :math:`k_B`.

    **kwargs
        Keyword arguments that are passed and are documented in
        ``sklearn.linear_model.LinearRegression``.

    Attributes
    ----------
    activation_energy_ : float
        Activation energy of the process, this is the same as
        ``-self.constant * self.reg_.coef_[0]``.

    extrapolated_process_ : float
        The extrapolation at room temperature of the thermally-induced
        process, note that this is the same that
        ``self.predict(np.array([[300.0]]))[0]``.

    reg_ : sklearn.linear_model.LinearRegressor
        The linear regressor for :math:`\ln k` versus :math:`\frac{1}{T}`.
    """

    def __init__(self, constant, **kwargs):
        self.constant = constant
        self.reg_ = sklearn.linear_model.LinearRegression(**kwargs)

    def fit(self, X, y, sample_weight=None):
        """Fit the Arrhenius empirical equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Temperature data.

        y : array-like of shape (n_samples,)
            Target values of the thermally-induced process.

        sample_weight : array-like of shape (n_samples,), defualt=None
            Individual weight of each thermally-induced value.
        """
        self._X = 1 / X
        self._y = np.log(y)
        self._sample_weight = (
            sample_weight / y if sample_weight is not None else None
        )

        self.reg_.fit(self._X, self._y, self._sample_weight)

        self.activation_energy_ = -self.constant * self.reg_.coef_[0]
        self.extrapolated_process_ = self.predict(np.array([[300.0]]))[0]

        return self

    def predict(self, X):
        """Predict the thermally-induced process values.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Temperature data.
        """
        return np.exp(self.reg_.predict(1 / X))

    def to_dataframe(self, X, y, sample_weight=None):
        """Convert the data with the predictions to a ``pandas.DataFrame``.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Temperature data.

        y : array-like of shape (n_samples,)
            Target values of the thermally-induced process.

        sample_weight : array-like of shape (n_samples,), defualt=None
            Individual weight of each thermally-induced value.

        Returns
        -------
        pandas.DataFrame
            A ``pandas.DataFrame`` with the data.
        """
        df = pd.DataFrame(
            {
                "temperatures": X.ravel(),
                "reaction_rate": y,
                "reaction_rate_pred": self.predict(X),
            }
        )

        if sample_weight is not None:
            df["weights"] = sample_weight

        return df

    @property
    def plot(self):
        """Arrhenius plot accessor."""
        return ArrheniusPlotter(self)


class ArrheniusPlotter:
    """Arrhenius plot.

    Parameters
    ----------
    areg : sierras.ArrheniusRegressor
        An ArrheniusRegressor already fitted.
    """

    def __init__(self, areg):
        self.areg = areg

    def arrhenius(self, X, y, ax=None, data_kws=None, pred_kws=None):
        """Arrhenius plot function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The current axes.

        data_kws : dict, default=None
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.errorbar`` for the data points.

        pred_kws : dict, default=None
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.plot`` for the predictions values.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The current axes.
        """
        ax = plt.gca() if ax is None else ax

        data_kws = {} if data_kws is None else data_kws
        pred_kws = {} if pred_kws is None else pred_kws

        for key, value in zip(("marker", "ls", "label"), ("o", "", "data")):
            data_kws.setdefault(key, value)

        pred_kws.setdefault("label", "fit")

        ax.errorbar(
            self.areg._X,
            self.areg._y,
            yerr=self.areg._sample_weight,
            **data_kws,
        )
        ax.plot(self.areg._X, self.areg.reg_.predict(self.areg._X), **pred_kws)

        return ax
