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

import json
from os import path

mydir = path.dirname(path.abspath(__file__))

# Magnitude constants
with open(path.join(mydir, 'data/magnitude_parameters.json'), 'r') as jsonfile:
    magnitude = json.load(jsonfile)

# Magnitude constants
with open(path.join(mydir, 'data/magnitude_exceptions.json'), 'r') as jsonfile:
    mag_exceptions = json.load(jsonfile)

# Seconds per day and Sol
SEC_PER_DAY_EARTH = 86400
SEC_PER_DAY_MARS = 88775.2440
