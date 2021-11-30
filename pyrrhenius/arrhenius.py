#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt

import sklearn.linear_model


def arrhenius(temperatures, dcoeff, plot=True, write_output=True):
    """Arrhenius: Pasar a clase.
    Recibe:
      un np.array con las temperaturas y un diccionario con las temperaturas
      como claves y pendientes del msd como valores, opcionalmente se le puede
      pedir que plotee y guarde en 'difusion.png'
    Devuelve:
      el valor del coeficiente de difusión extrapolado a temperatura ambiente
      con su error calculado a través de propagación de errores y considerando
      la barra de error de cada punto (aproximada por la desviación estandar
      del promedio) en el ajuste.
    """
    a2cm = 1e-8 ** 2
    fs2s = 1e-15
    units_change = a2cm / fs2s

    tempinv = 1.0 / temperatures
    diffli, difflierr = [], []
    for temp in temperatures:
        d = dcoeff[f"{temp}"].mean()
        errd = dcoeff[f"{temp}"].std()

        diffli.append(np.log(units_change * d))
        difflierr.append(errd / d)
    diffli, difflierr = np.array(diffli), np.array(difflierr)

    reg = sklearn.linear_model.LinearRegression().fit(
        tempinv.reshape(-1, 1), diffli, sample_weight=difflierr
    )

    # extrapolacion y propagación de errores
    den = (
        np.sum((tempinv / difflierr) ** 2) * np.sum(1 / difflierr ** 2)
        - np.sum(tempinv / difflierr ** 2) ** 2
    )
    erra = np.sqrt(np.sum(1 / difflierr ** 2) / den)
    errb = np.sqrt(np.sum((tempinv / difflierr) ** 2) / den)

    d_amb = np.exp(reg.coef_[0] / 300 + reg.intercept_)
    d_amb_err = d_amb * np.sqrt((erra / 300) ** 2 + errb ** 2)

    if plot:
        plt.xlabel(r"1 / T [K$^{-1}$]")
        plt.ylabel(r"ln D [cm$^2$ / s]")
        plt.ticklabel_format(style="sci", scilimits=(0, 0))
        plt.errorbar(
            tempinv, diffli, yerr=difflierr, marker="o", ls="", label="difusion"
        )
        plt.plot(tempinv, reg.coef_[0] * tempinv + reg.intercept_, label="ajuste")
        plt.legend()
        plt.savefig("arrhenius.png", dpi=600)
        plt.show()

    if write_output:
        with open("arrhenius.dat", "w") as f_arr:
            f_arr.write(
                f"# log(diff) = {reg.coef_[0]:.6e} / T + {reg.intercept_:.6e}\n"
            )
            f_arr.write(f"# extrapola a: {d_amb:.6e} +/- {d_amb_err:.6e}\n")
            f_arr.write(f"# tempinv, log(diff), log(differr)\n")
            for t, d, de in zip(tempinv, diffli, difflierr):
                f_arr.write(f"{t:.6e} {d:.6e} {de:.6e}\n")

    return d_amb, d_amb_err
