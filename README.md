# sierras

[![Github Actions CI](https://github.com/fernandezfran/sierras/actions/workflows/sierras_ci.yml/badge.svg)](https://github.com/fernandezfran/sierras/actions/workflows/sierras_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/sierras/badge/?version=latest)](https://sierras.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/sierras)](https://pypi.org/project/sierras/)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

**sierras** is a python module that allows to perform arrhenius plots for
diffusion coefficients and extrapolate to desired temperatures.


## Motivation

Obtaining room temperature trace diffusion coefficients from computational 
simulations exceeds the reasonable computational time that a simulation may 
require. Therefore, it is necessary to measure at different high temperatures, 
where diffusive processes are favored, to fit through an Arrhenius equation 
type and extrapolate to room temperature.


## Requirements

You need Python 3.9+ to run sierras.


## Installation

### Stable release

To install the most recent stable release of sierras with [pip](https://pip.pypa.io/en/stable/), 
run the following command in your termninal:

```bash
pip install sierras
```

### From sources

To installing it from sources you can clone this [GitHub repo](https://github.com/fernandezfran/sierras) 

```bash
git clone https://github.com/fernandezfran/sierras.git
```

and inside your local directory install it in the following way 

```bash
pip install -e .
```

## Usage

To start using sierras you can read the [tutorial](https://sierras.readthedocs.io/en/latest/tutorial.html) 
and the [API](https://sierras.readthedocs.io/en/latest/api.html).


## License

[MIT License](https://github.com/fernandezfran/sierras/blob/master/LICENSE)


### Contact info

<fernandezfrancisco2195@gmail.com>
