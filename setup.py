#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python tools to create products/plots for the final version of the InSight
Marsquake service Mars event catalogue

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2018
    Martin van Driel (Martin@vanDriel.de), 2018
    Luca Scarabello (luca.scarabello@sed.ethz.ch), 2024
    Savas Ceylan (savas.ceylan@eaps.ethz.ch), 2024
    Fabian Euchner (fabian.euchner@sed.ethz.ch), 2024
:license:
    GPLv3
"""

from setuptools import setup, find_packages

setup(name='final_catalog_products',
      version='0.9',
      description='Some python tools to create reports from the MQS database.',
      url='https://github.com/marsquakeservice/final_catalog_products',
      author='Simon Stähler, Martin van Driel, Luca Scarabello, Savas Ceylan, Fabian Euchner',
      author_email='fabian.euchner@sed.ethz.ch',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
            'numpy>2', 'obspy>=1.4.2', 'scipy', 'matplotlib', 'pandas', 'lxml',
            'plotly',   'tqdm', 'seaborn'])
