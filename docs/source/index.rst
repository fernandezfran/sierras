.. sierras documentation master file, created by
   sphinx-quickstart on Tue Dec  7 14:32:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======
sierras
=======

.. image:: https://github.com/fernandezfran/sierras/actions/workflows/sierras_ci.yml/badge.svg
   :target: https://github.com/fernandezfran/sierras/actions/workflows/sierras_ci.yml
   :alt: GitHub Actions CI

.. image:: https://readthedocs.org/projects/sierras/badge/?version=latest
   :target: https://sierras.readthedocs.io/
   :alt: ReadTheDocs

.. image:: https://img.shields.io/pypi/v/sierras
   :target: https://pypi.org/project/sierras/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00                  
   :target: https://github.com/leliel12/diseno_sci_sfw                             
   :alt: Curso doctoral FAMAF: Diseño de software para cómputo científico


**sierras** is a Python package to fit the arrhenius equation to diffusion 
coeffcients data, extrapolate to room temperature and plot

Motivation
----------

Obtaining room temperature trace diffusion coefficients from computational 
simulations exceeds the reasonable computational time that a simulation may 
require. Therefore, it is necessary to measure at different high temperatures, 
where diffusive processes are favored, to fit through an Arrhenius equation 
type and extrapolate to room temperature.


Requirements
------------

You need Python 3.9+ to run sierras.


| **Authors**
| Francisco Fernandez (E-mail: fernandezfrancisco2195@gmail.com)


Repository
----------

https://github.com/fernandezfran/sierras/


License
-------

sierras is under `MIT License <https://github.com/fernandezfran/sierras/blob/master/LICENSE>`__


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorial.ipynb
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
