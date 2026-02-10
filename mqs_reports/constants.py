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

import numpy as np 

from obspy import UTCDateTime


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

## date/time 

TIMESTAMP_SOL0 = UTCDateTime(2018, 11, 26, 5, 10, 50.33508)


## sensor characteristics
            
SENSOR_DIRECTIONS = {
    'dip': dict(dip_u=-29.4, dip_v=-29.2, dip_w=-29.7),
    'azimuth': dict(azi_u=135.1, azi_v=15.0, azi_w=255.0)
}

CHANNEL_ZNE_CODES = ('Z', 'N', 'E')


## defaults for waveform reading

DEFAULT_STATION_NAME = "ELYSE"
DEFAULT_LOCATION_CODE = "00"

# RAW, DEGLITCHED, DENOISED
DEFAULT_WAVFORM_TYPE = "RAW"

# DISP (displacement), 'VEL' (velocity), 'ACC' (acceleration)
DEFAULT_WAVFORM_KIND = "DISP"

WAVEFORM_READ_SP_FMIN = 0.5
WAVEFORM_READ_VBB_FMIN = 1.0 / 30.0
WAVEFORM_READ_T_PAD_VBB = 300.0

WAVEFORM_READ_SP_T_PRE = 100.0
WAVEFORM_READ_VBB_T_PRE = 1200.0

WAVEFORM_READ_INITIAL_START_END_TIME_MARGIN = 300.0
WAVEFORM_READ_INITIAL_NEAREST_SAMPLE = False

WAVEFORM_READ_TIME_MARGIN_MHZ = 900.0

WAVEFORM_READ_MHZ_PRE_FILT_FMIN_1 = 0.005
WAVEFORM_READ_MHZ_PRE_FILT_FMIN_2 = 0.01

WAVEFORM_READ_SUBSAMPLE_SHIFT_CORRECTION_BEFORE = "20190418T12:24"

WAVEFORM_READ_RESPONSE_FILTERED_TIME_MARGIN = 7 * 86400.0
WAVEFORM_READ_RESPONSE_LASTEPOCH_TIME_MARGIN = 1.0

# trace rotation

WAVEFORM_READ_ROTATE_NEAREST_SAMPLE = True


# filter trace on reading 
# - (1) ObsPy detrend/demean
# - (2) ObsPy taper(0.1)
# - (3) ObsPy filter highpass, zerophase, 0.5 * fmin (fmin=0.05)
# - (4) ObsPy detrend/default

# ObsPy detrend default: 'simple'
WAVEFORM_READ_FILTER_DETREND_SIMPLE = 'simple'
WAVEFORM_READ_FILTER_DETREND_DEMEAN = 'demean'

WAVEFORM_READ_TAPER_MAX_PERCENTAGE = 0.1

# ObsPy taper default: 'hann'
WAVEFORM_READ_TAPER_TYPE = 'hann' 

# ObsPy Butterworth highpass
WAVEFORM_READ_FILTER_BAND_TYPE = 'highpass'
WAVEFORM_READ_FILTER_FREQ_TYPE = 'butter'

WAVEFORM_READ_FILTER_HIGHPASS_FMIN = 0.05

# zero phase filter
WAVEFORM_READ_FILTER_HIGHPASS_ZP = True
        
## defaults for spectra computation 

SPECTRA_WELSH_WINDOW_LENGTH_SEC = 20.0
SPECTRA_DETICK_NUMBER_SAMPLES = 10
SPECTRA_DETICK_NUMBER_SAMPLES_DEGLITCHED = 0

SPECTRA_ZEROPAD_SIGNAL = True


## defaults for filterbank computation 

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

VBB_CHANNEL_MASK = {'VBB_Z': '??Z', 'VBB_N': '??N', 'VBB_E': '??E'}
VBB100_CHANNEL_MASK = {'VBB100_Z': '??Z', 'VBB100_N': '??N',  'VBB100_E': '??E'}
SP_CHANNEL_MASK = {'SP_Z': '??Z',  'SP_N': '??N',  'SP_E': '??E'}

WF_CHANNEL_CONFIG = {
    "waveforms_VBB": VBB_CHANNEL_MASK,
    "waveforms_VBB100": VBB100_CHANNEL_MASK,
    "waveforms_SP": SP_CHANNEL_MASK}
    

## default plot parameters

# spectra 




# filterbanks 

PLOT_FILTERBANK_FMAX_LF = 8.0
PLOT_FILTERBANK_FMIN_LF = 1.0 / 32.0

PLOT_FILTERBANK_FMAX_HF = 16.0
PLOT_FILTERBANK_FMIN_HF = 1.0 / 2.0

PLOT_FILTERBANK_DF_LF = 2.0**0.5
PLOT_FILTERBANK_DF_HF = 2.0**0.25


# SF, DL
PLOT_FILTERBANK_FMIN_SP_DL = 0.5
PLOT_FILTERBANK_FMAX_SP_DL = 32.0 * np.sqrt(2.0)
            
            
# VF, both
PLOT_FILTERBANK_FMIN_VF_BOTH = 1.0 / 8.0
PLOT_FILTERBANK_FMAX_VF_BOTH = 32.0 * np.sqrt(2.0)
                

# VF, SP
PLOT_FILTERBANK_FMIN_VF_SP = 1.0 / 8.0
PLOT_FILTERBANK_FMAX_VF_SP = 10.0


