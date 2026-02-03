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

# use values from processing-tools/constants.py
CRUST_VP = 4.0
CRUST_VS = 4.0 / (3.0**0.5)

DEFAULT_STATION_NAME = "ELYSE"
DEFAULT_LOCATION_CODE = "00"

# RAW, DEGLITCHED, DENOISED
DEFAULT_WAVFORM_TYPE = "RAW"

# DISP (displacement), 'VEL' (velocity), 'ACC' (acceleration)
DEFAULT_WAVFORM_KIND = "DISP"

WAVEFORM_READ_SP_FMIN = 0.5
WAVEFORM_READ_VBB_FMIN = 1.0 / 30.0
WAVEFORM_READ_T_PAD_VBB = 300.0


## default path names

PICKLE_EXTENSION = "pickle"
JSON_EXTENSION = "json"

EVENT_PRE_COMPUTE_DIR = "./events/"

PRE_COMPUTE_WAVEFORM_DIR = "waveforms/"
PRE_COMPUTE_METADATA_DIR = "metadata/"
PRE_COMPUTE_SPECTRA_DIR = "spectra/"
PRE_COMPUTE_FILTERBANK_DIR = "filterbanks/"

METADATA_CATEGORY_EVENT = "event"
METADATA_CATEGORY_SPECTRA = "spectra"
METADATA_CATEGORY_FILTERBANK = "filterbanks"

METADATA_SUB_CATEGORY_DATA = "data"
METADATA_SUB_CATEGORY_DPLOT = "plot"

DEFAULT_JSON_INPUT_FILE = "catalog.json"
EVENT_METADATA_PRE_COMPUTE_JSON_FILE = "{}_metadata.json"

VBB_FILE_TEMPLATE = "waveforms_VBB_{}.mseed"
VBB100_FILE_TEMPLATE = "waveforms_VBB100_{}.mseed"
SP_FILE_TEMPLATE = "waveforms_SP_{}.mseed"



