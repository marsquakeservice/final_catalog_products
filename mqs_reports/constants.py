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

QUALITY_FOR_FILTERBANK = ('A', 'B', 'C')

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
    

## default plot processing parameters

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


PLOT_FILTERBANK_ENVELOPE_WINDOW_SECONDS = 10.0
PLOT_FILTERBANK_ANNOTATION_TIME_MARGIN_SEC = 180.0


PLOT_FILTERBANK_START_END_TIME_MARGIN_OUT_SEC = 300.0

PLOT_FILTERBANK_START_TIME_MARGIN_IN_SEC = 300.0
PLOT_FILTERBANK_END_TIME_MARGIN_IN_SEC = 1100.0
PLOT_FILTERBANK_TIME_MIN_PLOT_MARGIN_IN_SEC = -240.0
PLOT_FILTERBANK_TIME_MAX_PLOT_MARGIN_IN_SEC = 900.0


PLOT_FILTERBANK_START_TIME_MARGIN_PHASE_SEC = 120.0
PLOT_FILTERBANK_END_TIME_MARGIN_PHASE_SEC = 240.0
PLOT_FILTERBANK_TIME_MIN_PLOT_MARGIN_PHASE_SEC = -50.0
PLOT_FILTERBANK_TIME_MAX_PLOT_MARGIN_PHASE_SEC = 200.0

# should not be needed
PLOT_FILTERBANK_START_END_TIME_MARGIN_FALLBACK_SEC = 100.0


## default plotting parameters

DEFAULT_FIGURE_DPI = 200.0

# filterbank and rotation plots
PLOT_MARKER_FOR_PHASES = ('P', 'S', 'x1', 'x2', 'x3', 'PP', 'SS')


# spectra

SPECTRA_FIGURE_SIZE = (20, 12)
SPECTRA_FIGURE_GRIDSPEC = (2, 2)

SPECTRA_FIGURE_PAR = dict(
    nrows=1, ncols=1, sharex='all', sharey='all', 
    figsize=SPECTRA_FIGURE_SIZE)

SPECTRA_FIGURE_POSITIONS = dict(
    top=0.911, bottom=0.097, left=0.049, right=0.972, hspace=0.2, wspace=0.116)


# original seaborn color_palette() used in older versions
# index 0 blue, 1 orange, 2 green, 3 dark red, 7 darkgray, 9 turquoise

# Paolo Veronese green
COLOR_SPECTRA_FILTERED_TRACE = "#009b7d"

# old silver (grey), darker than silver chalice
COLOR_SPECTRA_NOISE = "#848482"

# silver chalice (grey), lighter than old silver
COLOR_SPECTRA_NOISE_HIGH_SPS = "#acacac"

COLOR_SPECTRA_P_PHASE = "red"
COLOR_SPECTRA_S_PHASE = "blue"

# chartreuse
COLOR_SPECTRA_MANUAL_FIT = "#7fff00"

# sinopia (orange), darker than coral
COLOR_SPECTRA_PHASE_P_LF = "#cb410b"

# coral (orange), lighter than sinopia
COLOR_SPECTRA_PHASE_P_HIGH_SPS = "#ff7f50"

# new car (blue), darker than medium sky blue
COLOR_SPECTRA_PHASE_S_LF = "#214fc6"

# medium sky blue (blue), lighter than new car
COLOR_SPECTRA_PHASE_S_HIGH_SPS = "#80daeb"

COLOR_SPECTRA_TOP_OT = "black"

COLOR_SPECTRA_DICT = {
    "phase_p_lf": COLOR_SPECTRA_PHASE_P_LF, 
    "phase_s_lf": COLOR_SPECTRA_PHASE_S_LF, 
    "manual_fit": COLOR_SPECTRA_MANUAL_FIT, 
    "phase_s": COLOR_SPECTRA_S_PHASE, 
    "phase_p_high_sps": COLOR_SPECTRA_PHASE_P_HIGH_SPS,
    "phase_s_high_sps": COLOR_SPECTRA_PHASE_S_HIGH_SPS,
    "noise": COLOR_SPECTRA_NOISE, 
    "noise_high_sps": COLOR_SPECTRA_NOISE_HIGH_SPS, 
    "phase_p": COLOR_SPECTRA_P_PHASE}


SPECTRA_PLOT_TOP_OT_LABEL = 'OT'
SPECTRA_PLOT_TOP_OT_MARKER_ALPHA = 0.3
SPECTRA_PLOT_TOP_OT_LABEL_ALPHA = 0.5


COLOR_SPECTRA_BOTTOM_FREQ_BOX = "darkgrey"
COLOR_SPECTRA_BOTTOM_A0 = "cornflowerblue"
COLOR_SPECTRA_BOTTOM_F_CENTER = "crimson"

# Davy's grey
COLOR_SPECTRA_TOP_TEXT_BOXES = "#555555"

SPECTRA_TEXT_BOXES_XCOORD = {
    'type_quality': 0.05,
    'origin_time': 0.15,
    'raw_denoised_deglitched': 0.35,
    'streamid_lf': 0.6,
    'streamid_hf': 0.75,
    'filtercode': 0.85}

SPECTRA_TEXT_BOXES_YCOORD = 1.05
SPECTRA_TEXT_BOXES_PADDING = 0.2
SPECTRA_TEXT_BOXES_FACECOLOR = "white"
SPECTRA_TEXT_BOXES_ALPHA = 0.5
SPECTRA_TEXT_BOXES_FONTSIZE = 15


SPECTRA_TEXT_BOX_PARAMS = {
    'boxstyle': 'square', 
    'facecolor': 'white', 
    'edgecolor': COLOR_SPECTRA_TOP_TEXT_BOXES, 
    'pad': SPECTRA_TEXT_BOXES_PADDING}

SPECTRA_PLOT_TOP_XLABEL = "Time [seconds]"
SPECTRA_PLOT_TOP_XLABEL_TEMPLATE = "Time after {} [seconds]"

SPECTRA_PLOT_TOP_YLABEL = "Displacement ({}) [m]"

SPECTRA_PLOT_BOTTOM_XLABEL = "Frequency [Hz]"
SPECTRA_PLOT_BOTTOM_YLABEL = "Displacement PSD [dB]"

SPECTRA_PLOT_BOTTOM_YAXIS_MIN_RANGE = 0.85

SPECTRA_PLOT_BOTTOM_FC_LABEL_OFFSET = 1.0
SPECTRA_PLOT_BOTTOM_A0_LABEL_OFFSET = 1.0

SPECTRA_PLOT_BOTTOM_FC_MARKER_LINESTYLE = "dashed"
SPECTRA_PLOT_BOTTOM_FC_MARKER_ALPHA = 0.3
SPECTRA_PLOT_BOTTOM_FC_LABEL_ALPHA = 0.5

SPECTRA_PLOT_BOTTOM_A0_MARKER_LINESTYLE = "dashed"
SPECTRA_PLOT_BOTTOM_A0_MARKER_ALPHA = 0.3
SPECTRA_PLOT_BOTTOM_A0_LABEL_ALPHA = 0.5


SPECTRA_PLOT_FILTER_BOX_LABEL_1 = r'$\mathrm{filter:\,HP_{BW}^{ZP}('
SPECTRA_PLOT_FILTER_BOX_LABEL_2 = r'\,Hz)}$'

SPECTRA_PLOT_SUPTITLE_FONTSIZE = 'x-large'


# filterbanks 

# 10, 16
FILTERBANK_FIGURE_SIZE = (20, 16)
FILTERBANK_FIGURE_GRIDSPEC = (2, 3)

FILTERBANK_FIGURE_GRIDSPEC_HEIGHT_RATIOS = (1, 32)

FILTERBANK_FIGURE_PAR = dict(nrows=2, ncols=3, sharex='all', sharey='all')

FILTERBANK_FIGURE_DPI = DEFAULT_FIGURE_DPI 

FILTERBANK_FIGURE_POSITIONS = dict(
    top=0.911, bottom=0.097, left=0.089, right=0.972, hspace=0.05, wspace=0.116)

# original seaborn color_palette() used in older versions
# index 0 blue, 1 orange, 2 green, 3 dark red, 7 darkgray, 9 turquoise

# Paolo Veronese green
# COLOR_SPECTRA_FILTERED_TRACE = "#009b7d"

# old silver (grey), darker than silver chalice
# COLOR_SPECTRA_NOISE = "#848482"

# silver chalice (grey), lighter than old silver
# COLOR_SPECTRA_NOISE_HIGH_SPS = "#acacac"

COLOR_FILTERBANK_P_PHASE = "red"
COLOR_FILTERBANK_S_PHASE = "blue"
COLOR_FILTERBANK_START_END_PHASE = "darkgreen"
COLOR_FILTERBANK_GRID_HORIZONTAL = 'black'

# Davy's grey
COLOR_FILTERBANK_TOP_TEXT_BOXES = "#555555"

FILTERBANK_PLOT_PHASES_MARKER_LINESTYLE = "dashed"
FILTERBANK_PLOT_PHASES_MARKER_LINEWIDTH = 0.5
FILTERBANK_PLOT_PHASES_MARKER_ALPHA = 0.3

FILTERBANK_PLOT_PHASES_MARKER_Y_OFFSET = -0.5 
FILTERBANK_PLOT_PHASES_MARKER_X_ALIGN = "left"

FILTERBANK_PLOT_GLITCH_YMIN_DEFAULT = -2.0
FILTERBANK_PLOT_GLITCH_HEIGHT_DEFAULT = 50.0 

FILTERBANK_PLOT_GLITCH_YMIN = -1.0
FILTERBANK_PLOT_GLITCH_HEIGHT = 0.3

FILTERBANK_PLOT_GLITCH_FONTCOLOR = 'grey'
FILTERBANK_PLOT_GLITCH_ANNOTATION_FONTCOLOR = 'lightgrey'
FILTERBANK_PLOT_GLITCH_ANNOTATION_ZORDER = -3
                        
FILTERBANK_PLOT_GLITCH_ALPHA = 0.8

FILTERBANK_PLOT_ENVELOPE_LOG_LINEWIDTH = 1.0
FILTERBANK_PLOT_ENVELOPE_LOG_ZORDER = 50 

FILTERBANK_PLOT_ENVELOPE_LIN_LINEWIDTH = 1.0 
FILTERBANK_PLOT_ENVELOPE_LIN_ZORDER = 50 

FILTERBANK_PLOT_GRID_HORIZONTAL_LINESTYLE = 'dashed'
FILTERBANK_PLOT_GRID_HORIZONTAL_LINEWIDTH = 1.0
FILTERBANK_PLOT_GRID_HORIZONTAL_ALPHA = 0.7

FILTERBANK_PLOT_GRID_LINEWIDTH = 0.6
FILTERBANK_PLOT_GRID_ALPHA = 0.5

FILTERBANK_PLOT_GRID_XTICKS = np.arange(-300, 3000, 25)

FILTERBANK_TEXT_BOXES_XCOORD = {
    'type_quality': 0.05,
    'origin_time': 0.165,
    'p_phase_time': 0.325,
    'raw_denoised_deglitched': 0.475,
    'streamid_lf': 0.625,
    'streamid_hf': 0.774,
    'freq_range': 0.925}

FILTERBANK_TEXT_BOXES_YCOORD = 1.05
FILTERBANK_TEXT_BOXES_PADDING = 0.2
FILTERBANK_TEXT_BOXES_FACECOLOR = "white"
FILTERBANK_TEXT_BOXES_FONTSIZE = 15

FILTERBANK_TEXT_BOX_PARAMS = {
    'boxstyle': 'square', 
    'facecolor': 'white', 
    'edgecolor': COLOR_FILTERBANK_TOP_TEXT_BOXES, 
    'pad': FILTERBANK_TEXT_BOXES_PADDING}


FILTERBANK_PLOT_PHASES_XLABEL_TEMPLATE = "Time after {} arrival ({}) [s]"
FILTERBANK_PLOT_NO_PHASES_XLABEL_TEMPLATE = "Time after start ({}) [s]"

FILTERBANK_PLOT_XTICKLABEL_FONTSIZE = 20
FILTERBANK_PLOT_PHASES_XLABEL_FONTSIZE = 20
FILTERBANK_PLOT_PHASES_XLABEL_PAD = 15

FILTERBANK_PLOT_YLABEL = "Frequency [Hz]"
FILTERBANK_PLOT_YLABEL_FONTSIZE = 20
FILTERBANK_PLOT_YTICKLABEL_FONTSIZE = 20

FILTERBANK_PLOT_TITLE_FONTSIZE = 20

# fontsize 24 is too large 
FILTERBANK_PLOT_SUPTITLE_FONTSIZE = 'x-large'

FILTERBANK_PLOT_PHASE_LABEL_OFFSET = 1.0
