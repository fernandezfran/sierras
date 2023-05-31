# sierras

[![Github Actions CI](https://github.com/fernandezfran/sierras/actions/workflows/sierras_ci.yml/badge.svg)](https://github.com/fernandezfran/sierras/actions/workflows/sierras_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/sierras/badge/?version=latest)](https://sierras.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/sierras)](https://pypi.org/project/sierras/)
[![python version](https://img.shields.io/badge/python-3.8%2B-4584b6)](https://www.python.org/)
[![mit license](https://img.shields.io/badge/License-MIT-ffde57)](https://github.com/fernandezfran/sierras/blob/main/LICENSE)
[![downloads](https://static.pepy.tech/badge/sierras)](https://pepy.tech/project/sierras)
[![diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

**sierras** is a tool for empirical Arrhenius equation fitting for 
thermally-induced physicochemical processes.


## Requirements

You need Python 3.8+ to run sierras.


## Installation

You can install the most recent stable release of sierras with 
[pip](https://pip.pypa.io/en/latest/)

```
python -m pip install -U pip
python -m pip install -U sierras
```


## Usage

A simple example of use:

```python
import sierras

k_boltzmann = 8.617333262e-5  # eV / K
areg = sierras.ArrheniusRegressor(k_boltzmann)

areg.fit(Temperatures, target_process)

areg.activation_energy_  # in this case in eV
areg.extrapolated_process_  # extrapolated process at room temperature

areg.plot.arrhenius(Temperatures, target_process)  # plot the fitting
```


For a more detailed explanation you can read the 
[tutorial](https://sierras.readthedocs.io/en/latest/tutorial.html) 
and the [API](https://sierras.readthedocs.io/en/latest/api.html).


## License

[MIT License](https://github.com/fernandezfran/sierras/blob/master/LICENSE)


## Contact info

You can contact me at <ffernandev@gmail.com>
