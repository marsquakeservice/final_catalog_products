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
    GPL
"""

import copy
import io
import os.path
import pickle
import sys
import warnings

from os import makedirs
from os.path import exists as pexists, join as pjoin

from sys import stdout as stdout

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from obspy import UTCDateTime
from obspy import UTCDateTime as utct 

from tqdm import tqdm

from fittingparam import FittingParameterPool

import mqs_reports.constants as constants

from mqs_reports.annotations import Annotations
from mqs_reports.constants import PICKLE_EXTENSION
from mqs_reports.event import ENVELOPE_CORNERS_COUNT, \
    FILTERBANK_CORNERS_COUNT, FILTERBANK_PLOT_SCALE_FACTOR, PICK_METHOD_ALIGNED

from mqs_reports.utils import add_orientation_to_stream_info
from mqs_reports.utils import envelope_smooth
from mqs_reports.utils import uncertainty_from_pdf

# from marsprocessingtools import catalog as marscatalog
from marsprocessingtools import utils as marsutils

sns.set_theme(style="darkgrid")

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

COLOR_SPECTRA_TOP_OT = "black"

COLOR_SPECTRA_BOTTOM_FREQ_BOX = "darkgrey"
COLOR_SPECTRA_BOTTOM_A0 = "cornflowerblue"
COLOR_SPECTRA_BOTTOM_F_CENTER = "crimson"

# Davy's grey
COLOR_SPECTRA_TOP_TEXT_BOXES = "#555555"

SPECTRA_FIGURE_SIZE = (20, 12)
SPECTRA_FIGURE_POSITIONS = dict(
    top=0.911, bottom=0.097, left=0.049, right=0.972, hspace=0.2, 
    wspace=0.116)
            
SPECTRA_TEXT_BOXES_XCOORD = {
    'type_quality': 0.05,
    'origin_time': 0.15,
    'raw_denoised_deglitched': 0.30,
    'streamid_lf': 0.5,
    'streamid_hf': 0.7,
    'filtercode': 0.85}

SPECTRA_TEXT_BOXES_YCOORD = 1.05
SPECTRA_TEXT_BOXES_PADDING = 0.2
SPECTRA_TEXT_BOXES_FACECOLOR = "white"

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

SPECTRA_PLOT_BOTTOM_YAXIS_MIN_RANGE = 0.95

SPECTRA_PLOT_FILTER_BOX_LABEL_1 = r'$\mathrm{filter:\,HP_{BW}^{ZP}('
SPECTRA_PLOT_FILTER_BOX_LABEL_2 = r'\,Hz)}$'


def plot_spectra(
    fitter, 
    fitting_parameters, 
    fitting_parameters_defaults, 
    dir_out: str,
    wf_type: str="RAW", 
    rotate: bool=False, 
    smprate: str="",
    orientation: list=[], 
    force_products: bool=False, 
    calculate_spectra: bool=False, 
    plot_spectra: bool=True) -> None:
    
    if (not rotate and 'ZNE' not in orientation):
        print("products.plot_spectra, ZNE not requested")
        return 
        
    if (rotate and 'ZRT' not in orientation):
        print("products.plot_spectra, ZRT not requested")
        return 
        
    print("products: calculate/plot spectra for {} waveforms (smprate {}, "\
        "ZRT {})".format(wf_type, smprate, rotate))
    
    for event in tqdm(fitter.catalog, file=sys.stdout):
        
        if event.waveforms_VBB is None:
            print("products: event {}, no VBB waveforms exist, skipping".format(
                event.name))
            continue
        
        if rotate and event.baz is None:
            print("products: event {}, rotation to ZRT requested but no BAZ "\
                "exists, skipping".format(event.name))
            continue
        
        avail_rate = event.available_sampling_rates()
        
        if smprate == 'VBB_LF':
            if avail_rate['VBB_Z'] is None or \
               avail_rate['VBB_N'] is None or \
               avail_rate['VBB_E'] is None:
                continue
            instrument = 'VBB'
            
        elif smprate == 'SP_HF':
            if avail_rate['SP_Z'] != 100. or \
               avail_rate['SP_N'] != 100. or \
               avail_rate['SP_E'] != 100.:
                continue
            instrument = 'SP'
            
        elif smprate == 'LF+HF':
            if avail_rate['VBB100_Z'] == 100. and \
               avail_rate['VBB100_N'] == 100. and \
               avail_rate['VBB100_E'] == 100.:
                instrument = 'VBB+VBB100'
            
            elif avail_rate['SP_Z'] == 100. and \
                 avail_rate['SP_N'] == 100. and \
                 avail_rate['SP_E'] == 100.:
                instrument = 'VBB+SP'
            
            else:
                continue
        
        else:
            raise ValueError(f'Invalid value for smprate: {smprate}')
        
        if 'noise_start' in event.picks and \
            len(event.picks['noise_start']) > 0:
            noise_start = event.picks['noise_start']
        
        else:
            noise_start = event.picks['start']

        if 'noise_end' in event.picks and \
            len(event.picks['noise_end']) > 0:
            noise_end = event.picks['noise_end']
        else:
            noise_end = str(UTCDateTime(event.picks['start']) + 30)

        if 'P_spectral_start' in event.picks and \
            len(event.picks['P_spectral_start']) > 0:
            p_start = event.picks['P_spectral_start']
        else:
            p_start = str(UTCDateTime(event.picks['start']) + 30)

        if 'P_spectral_end' in event.picks and \
            len(event.picks['P_spectral_end']) > 0:
            p_end = event.picks['P_spectral_end']
        else:
            p_end = str(UTCDateTime(event.picks['start']) + 60)

        if 'S_spectral_start' in event.picks and \
            len(event.picks['S_spectral_start']) > 0:
            s_start = event.picks['S_spectral_start']
        else:
            s_start = str(UTCDateTime(event.picks['start']) + 90)

        if 'S_spectral_end' in event.picks and \
            len(event.picks['S_spectral_end']) > 0:
            s_end = event.picks['S_spectral_end']
        
        else:
            s_end = str(UTCDateTime(event.picks['start']) + 120)

        spectral_windows = {
                'noise_start': noise_start, 'noise_end': noise_end,
                'P_spectral_start': p_start, 'P_spectral_end': p_end,
                'S_spectral_start': s_start, 'S_spectral_end': s_end}

        print("products: swap events for event {}, {}/Q{}, wf {}, smprate "\
            "{}, ZRT {}".format(event.name, event.mars_event_type_short, 
                event.quality, event.wf_type, smprate, rotate))
        
        detick_nfsamp = (
            constants.SPECTRA_DETICK_NUMBER_SAMPLES if \
                wf_type != "DEGLITCHED" else \
                    constants.SPECTRA_DETICK_NUMBER_SAMPLES_DEGLITCHED)
        
        try:
            # implicitly calls fitter.calc_spectra()
            fitter.swap_event(
                event_name=event.name,
                instrument=instrument, 
                rotate=rotate,
                winlen_sec=constants.SPECTRA_WELSH_WINDOW_LENGTH_SEC, 
                detick_nfsamp=detick_nfsamp,
                time_windows=spectral_windows, 
                smprate=smprate, 
                force_products=force_products, 
                calculate_spectra=calculate_spectra, 
                keep_spectra=plot_spectra)
            
        except Exception as e:
            print(f"Error fitter.swap_event with event {event.name}: {e}")
            continue

        if not plot_spectra:
            continue
        
        # continue only if plot requested
        ev_folder = pjoin(dir_out, fitter.event.name)

        if not os.path.exists(ev_folder):
            os.makedirs(ev_folder)

        def plot_filename(ev, component):
            return pjoin(
                ev_folder,
                "spectra_{}_SampRate_{}_Component_{}_Data_{}.png".format(
                    ev.name, smprate, component, ev.wf_type))

        fitting_parameters_pool = FittingParameterPool(
            event_name=fitter.event.name)
        
        if fitter.event.name in fitting_parameters:
            fitting_parameters_pool.set_parameters(
                fitting_parameters[fitter.event.name])
            fitting_parameters_pool.set_value(
                None, 'is_manually_reviewed', True)
        
        else: 
            # Get the default values
            profiles = fitting_parameters_defaults['fitting-defaults']\
                ['event-settings'].split(',')
            defaults = None
            
            for profile in profiles:
                preset = profile.split(':')
                if fitter.get_event_type() == preset[0].strip():
                    defaults = copy.deepcopy(
                        fitting_parameters_defaults['fitting-defaults']\
                            [preset[1].strip()])
                    break
            
            if defaults is None:
                print("Error: no default fitting parameters found for event "\
                    "{}".format(fitter.event.name))
                continue
            
            fitting_parameters_pool.set_parameters(defaults)
            fitting_parameters_pool.set_value(
                None, 'is_manually_reviewed', False)

        # add missing info for component R and T
        fitting_parameters_pool.set_value(
            "R",'fminP',fitting_parameters_pool.get_value("Z", 'fminP'))
        fitting_parameters_pool.set_value(
            "R",'fmaxP',fitting_parameters_pool.get_value("Z", 'fmaxP'))
        fitting_parameters_pool.set_value(
            "R",'fminS',fitting_parameters_pool.get_value("Z", 'fminS'))
        fitting_parameters_pool.set_value(
            "R",'fmaxS',fitting_parameters_pool.get_value("Z", 'fmaxS'))

        fitting_parameters_pool.set_value("T",'fminP',
                min(fitting_parameters_pool.get_value("E", 'fminP'),
                    fitting_parameters_pool.get_value("N", 'fminP')))
        fitting_parameters_pool.set_value("T",'fmaxP',
                max(fitting_parameters_pool.get_value("E", 'fmaxP'),
                    fitting_parameters_pool.get_value("N", 'fmaxP')))
        fitting_parameters_pool.set_value("T",'fminS',
                min(fitting_parameters_pool.get_value("E", 'fminS'),
                    fitting_parameters_pool.get_value("N", 'fminS')))
        fitting_parameters_pool.set_value("T",'fmaxS',
                max(fitting_parameters_pool.get_value("E", 'fmaxS'),
                    fitting_parameters_pool.get_value("N", 'fmaxS')))

        # get stream from pre-computed waveforms
        if instrument == 'VBB':
            stream = fitter.event.waveforms_VBB.copy()
        elif instrument == 'SP':
            stream = fitter.event.waveforms_SP.copy()
        elif instrument == 'VBB100':
            stream = fitter.event.waveforms_VBB100.copy()
        elif instrument == 'VBB+VBB100':
            stream = fitter.event.waveforms_VBB100.copy()
        elif instrument == 'VBB+SP':
            stream = fitter.event.waveforms_SP.copy()
        else:
           raise ValueError(f'Invalid value for instrument: {instrument}')

        if rotate:
            if stream is not None:
                stream.rotate('NE->RT', back_azimuth=fitter.event.baz)

        LF_streaminfo = ""
        LF_streaminfo_with_orientation = ""
        HF_streaminfo = ""
        HF_streaminfo_with_orientation = ""
        
        if 'stream_info' in fitter.event.spectra and \
                fitter.event.spectra['stream_info'].startswith("LF"):
            LF_streaminfo = fitter.event.spectra['stream_info']
            
        if 'stream_info' in fitter.event.spectra_SP and \
                fitter.event.spectra_SP['stream_info'].startswith("HF"):
            HF_streaminfo = fitter.event.spectra_SP['stream_info']

        print("plotting spectra for event {}".format(event.name))
        
        # get trace for plotting for requested component (orientation)
        for component in (['R','T'] if rotate else ['Z','N','E']):
            
            channel_name = "{}{}".format('*', component)
            
            tr = stream.select(channel=channel_name)[0].copy()

            fnam = plot_filename(fitter.event, component)

            if pexists(fnam) and not(force_products):
                print("products.plot_spectra: plot file {} exists, "\
                    "skipping".format(fnam))
                continue

            try:
                results = fitter.fit_for_component(
                    fitting_parameters=fitting_parameters_pool, 
                    component=component)
                
            except Exception as e:
                print(f'Error fitter.fit_for_component with event '\
                    '{fitter.event.name} component {component}: {e}')
                continue

            print("products.plot_spectra: create figure for plot file {}".format(
                fnam))
            
            fig = plt.figure(figsize=SPECTRA_FIGURE_SIZE)
            fig.subplots_adjust(**SPECTRA_FIGURE_POSITIONS)
            
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
    
            streaminfo_plot = dict(
                LF=None, HF=None, event_name=event.name, wf_type=wf_type,
                origin_time_screen=event.origin_time_screen,
                origin_time=event.origin_time)
            
            if len(LF_streaminfo) > 0:
                LF_streaminfo_with_orientation = add_orientation_to_stream_info(
                    LF_streaminfo, component)
                streaminfo_plot["LF"] = LF_streaminfo
                streaminfo_plot["LF_orientation"] = \
                    LF_streaminfo_with_orientation
            
            if len(HF_streaminfo) > 0:
                HF_streaminfo_with_orientation = add_orientation_to_stream_info(
                    HF_streaminfo, component)
                streaminfo_plot["HF"] = HF_streaminfo
                streaminfo_plot["HF_orientation"] = \
                    HF_streaminfo_with_orientation
            
            fig.suptitle(
                "Event {}".format(fitter.event.name), weight='bold', 
                fontsize='x-large')
            
            _plot_spectra_top(
                fitter, ax1, tr, component, spectral_windows,
                fitting_parameters_pool, streaminfo_plot)
            
            _plot_spectra_bottom(
                ax2, ax3, fitter, streaminfo_plot, component,
                fitting_parameters_pool, results, wf_type)
            
            fig.savefig(fnam)
            plt.close(fig)

    print("products.plot_spectra: processing has ended")
    
    
def _plot_spectra_top(
    fitter, ax, tr, component, windows, fitting_parameters, streaminfo_plot):

    #if parameters.filter_apply:
    #    tr.filter('bandpass', freqmin=parameters.filter_min_freq, 
    #              freqmax=parameters.filter_max_freq, 
    #              zerophase=parameters.filter_zero_phase,
    #              corners=parameters.filter_order)
    #    tr.taper(max_length=60, max_percentage=0.3)
    
    # this is stream_info with orientation
    stream_info_orientation = f'{tr.id}@{tr.stats.sampling_rate}'
    
    stream_info_hflf = ""
    if streaminfo_plot["HF"] is not None:
        stream_info_hflf = "HF"
    elif streaminfo_plot["LF"] is not None:
        stream_info_hflf = "LF"
    
    starttime_screen = marsutils.get_rounded_timestamps(
        tr.stats.starttime).get('TIMESTAMP_READABLE_FORMAT')
    
    ax.set(xlabel=SPECTRA_PLOT_TOP_XLABEL_TEMPLATE.format(starttime_screen))
    ax.set(ylabel=SPECTRA_PLOT_TOP_YLABEL.format(stream_info_hflf))

    to_tr_time = lambda time_str: UTCDateTime(time_str) - tr.stats.starttime

    # get minimun/maximum from trace for noise, P, S boxes
    data_min = np.min(tr.data)
    data_max = np.max(tr.data)

    noise_start = to_tr_time(windows['noise_start'])
    noise_end   = to_tr_time(windows['noise_end'])
    p_start     = to_tr_time(windows['P_spectral_start'])
    p_end       = to_tr_time(windows['P_spectral_end'])
    s_start     = to_tr_time(windows['S_spectral_start'])
    s_end       = to_tr_time(windows['S_spectral_end'])

    origin_time_plot = to_tr_time(streaminfo_plot['origin_time'])
    
    # plot filtered trace
    # NOTE: axis range is automatic
    trace_filtered = ax.plot(
        tr.times(), tr.data, color=COLOR_SPECTRA_FILTERED_TRACE, linestyle='-',
        label="filtered trace")
    
    # noise box
    rect_noise = patches.Rectangle(
        xy=(noise_start, data_min), width=noise_end-noise_start, 
        height=data_max-data_min, linewidth=3, 
        edgecolor=COLOR_SPECTRA_NOISE, alpha=0.5, facecolor="none",
        label="noise window")
    
    ax.add_patch(rect_noise)

    # P phase box
    rect_p = patches.Rectangle(
        xy=(p_start, data_min), width=p_end-p_start, height=data_max-data_min,
        linewidth=3, edgecolor=COLOR_SPECTRA_P_PHASE, alpha=0.3, 
        facecolor="none", label="P phase window")
    
    ax.add_patch(rect_p)

    # S phase box
    rect_s = patches.Rectangle(
        xy=(s_start, data_min), width=s_end-s_start, height=data_max-data_min,
        linewidth=3, edgecolor=COLOR_SPECTRA_S_PHASE, alpha=0.3, 
        facecolor="none", label="S phase window")
    
    ax.add_patch(rect_s)

    # custom legend
    lgd = ax.legend(
        handles=[trace_filtered[0], rect_noise, rect_p, rect_s], 
        loc='upper right')
    
    # seismic phases
    # this defines the order of pick types. Clumsy!
    # NOTE: Pg and Sg have been retired
    try:
        if fitter.get_event_type() in ['LF', 'WB', 'BB']:
            p_phase = fitter.get_pick('P') or fitter.get_pick('PP') or \
                fitter.get_pick('P1') or fitter.get_pick('x1') or \
                    fitter.get_pick('y1') or fitter.get_pick('start')
            
            s_phase = fitter.get_pick('S') or fitter.get_pick('SS') or \
                fitter.get_pick('S1') or fitter.get_pick('x2') or \
                    fitter.get_pick('y2') or fitter.get_pick('start')
        else:
            p_phase = fitter.get_pick('P') or fitter.get_pick('PP') or \
                fitter.get_pick('P1') or fitter.get_pick('y1') or \
                    fitter.get_pick('x1') or fitter.get_pick('start')
            
            s_phase = fitter.get_pick('S') or fitter.get_pick('SS') or \
                fitter.get_pick('S1') or fitter.get_pick('y2') or \
                    fitter.get_pick('x2') or fitter.get_pick('start')
    except:
        p_phase = fitter.get_pick('P') or fitter.get_pick('P1') or \
            fitter.get_pick('x1') or fitter.get_pick('y1') or \
                fitter.get_pick('start')
            
        s_phase = fitter.get_pick('S') or fitter.get_pick('S1') or \
            fitter.get_pick('x2') or fitter.get_pick('y2') or \
                fitter.get_pick('start')


    # mark origin time and seismic phases 
    # (width is based on plot range, no uncertainty)
    width = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.005
    
    _add_origin_time_marker(ax, origin_time_plot)
    
    if p_phase:
        P = to_tr_time(p_phase)
        ax.axvspan(
            xmin=P-0.5*width, xmax=P+0.5*width, facecolor=COLOR_SPECTRA_P_PHASE, 
            alpha=0.3, label="P phase")
        
    if s_phase:
        S = to_tr_time(s_phase)
        ax.axvspan(
            xmin=S-0.5*width, xmax=S+0.5*width, facecolor=COLOR_SPECTRA_S_PHASE, 
            alpha=0.3, label="S phase")
    
    # top subtitle text boxes (type/quality, OT, wf type, LF, HF, filter)
    ax.text(SPECTRA_TEXT_BOXES_XCOORD['type_quality'], 
        SPECTRA_TEXT_BOXES_YCOORD, 
        "{}/Q{}".format(fitter.event.mars_event_type_short, 
        fitter.event.quality),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, bbox=SPECTRA_TEXT_BOX_PARAMS,
        color=COLOR_SPECTRA_TOP_TEXT_BOXES, fontsize=15)
    
    ax.text(SPECTRA_TEXT_BOXES_XCOORD['origin_time'], 
        SPECTRA_TEXT_BOXES_YCOORD, 
        "OT: {}".format(streaminfo_plot["origin_time_screen"]),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, bbox=SPECTRA_TEXT_BOX_PARAMS,
        color=COLOR_SPECTRA_TOP_TEXT_BOXES, fontsize=15)
    
    ax.text(SPECTRA_TEXT_BOXES_XCOORD['raw_denoised_deglitched'], 
        SPECTRA_TEXT_BOXES_YCOORD, 
        "{} waveforms".format(streaminfo_plot["wf_type"].lower()),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, bbox=SPECTRA_TEXT_BOX_PARAMS,
        color=COLOR_SPECTRA_TOP_TEXT_BOXES, fontsize=15)
        
    if streaminfo_plot["LF"] is not None:
        ax.text(SPECTRA_TEXT_BOXES_XCOORD['streamid_lf'], 
            SPECTRA_TEXT_BOXES_YCOORD, 
            "{}".format(streaminfo_plot["LF_orientation"]),
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes, bbox=SPECTRA_TEXT_BOX_PARAMS,
            color=COLOR_SPECTRA_TOP_TEXT_BOXES, fontsize=15)
    
    if streaminfo_plot["HF"] is not None:
        ax.text(SPECTRA_TEXT_BOXES_XCOORD['streamid_hf'], 
            SPECTRA_TEXT_BOXES_YCOORD, 
            "{}".format(streaminfo_plot["HF_orientation"]),
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes, bbox=SPECTRA_TEXT_BOX_PARAMS,
            color=COLOR_SPECTRA_TOP_TEXT_BOXES, fontsize=15)
    
    filterbox_text_sub = SPECTRA_PLOT_FILTER_BOX_LABEL_1 + \
        str(constants.WAVEFORM_READ_FILTER_HIGHPASS_FMIN) + \
        SPECTRA_PLOT_FILTER_BOX_LABEL_2
    
    ax.text(SPECTRA_TEXT_BOXES_XCOORD['filtercode'], SPECTRA_TEXT_BOXES_YCOORD, 
        filterbox_text_sub, verticalalignment='center', 
        horizontalalignment='center',
        transform=ax.transAxes, bbox=SPECTRA_TEXT_BOX_PARAMS,
        color=COLOR_SPECTRA_TOP_TEXT_BOXES, fontsize=15)
    

def _set_axis_limits_psd_denoised(
    axis, data, ymin_scale=SPECTRA_PLOT_BOTTOM_YAXIS_MIN_RANGE):
    """
    Set minimum of y axis for denoised PSD.
    """
    
    npts = len(data)
    npts_new = int((1.0 - ymin_scale) * npts)
    
    # set y minimum at ymin_scale (0-1) of x-axis
    if npts_new > 0:
        data_new = data[0:npts - int((1.0 - ymin_scale) * npts)]
        
        # add 10% to computed minimum
        ylim_min = 1.1 * np.min(data_new)
        
        axis.set_ylim(ymin=ylim_min)
    
    return axis 


def _set_axis_limits(axis, data, wf_type, xlog=False, ylog=False):
    """
    TODO(fab): general axis limits
    """
    
    # y axis limits
    ylim_min, ylim_max = axis.get_ylim()
    
    # set y axis minimum to 50% if DENOISED
    if wf_type == "DENOISED":
        
        if ylog:
            ylim_min = np.log10(0.5 * ylim_min)
            
        else:
            ylim_min = 0.5 * ylim_min
    
    axis.set_ylim(ymin=ylim_min)
    
    return axis 


def _plot_spectra_bottom(
    axP, axS, fitter, streaminfo_plot, component, fitting_parameters, results, 
    wf_type):

    # P phase subplot
    axP.set_title(f'P phase [{component}]')
    axP.set(
        xlabel=SPECTRA_PLOT_BOTTOM_XLABEL, ylabel=SPECTRA_PLOT_BOTTOM_YLABEL, 
        xscale='log')
    
    # S phase subplot
    axS.set_title(f'S phase [{component}]')
    axS.set(
        xlabel=SPECTRA_PLOT_BOTTOM_XLABEL, ylabel=SPECTRA_PLOT_BOTTOM_YLABEL,
        xscale='log')
    
    colors = COLOR_SPECTRA_DICT
    
    ## VBB
    if streaminfo_plot["LF"]:

        # noise is too low amplitude and affects the y-axis range
        if wf_type != "DENOISED":

            # Noise for P and S
            noise_psd = fitter.get_noise_spectrum(component)[1:]
            noise_freq = fitter.get_noise_frequency()[1:]

            sns.lineplot(
                ax=axP, x=noise_freq, y=noise_psd, label='noise', 
                color=colors["noise"])
            sns.lineplot(
                ax=axS, x=noise_freq, y=noise_psd, label='noise', 
                color=colors["noise"])

        # P
        P_psd = fitter.get_P_spectrum(component)[1:]
        P_freq = fitter.get_P_frequency()[1:]

        sns.lineplot(
            ax=axP, x=P_freq, y=P_psd, label='P phase (LF)', 
            color=colors["phase_p_lf"])

        # S
        S_psd = fitter.get_S_spectrum(component)[1:]
        S_freq = fitter.get_S_frequency()[1:]

        sns.lineplot(
            ax=axS, x=S_freq, y=S_psd, label='S phase (LF)', 
            color=colors["phase_s_lf"])

    ## SP
    if streaminfo_plot["HF"]:

        # noise is too low amplitude and affects the y-axis range
        if wf_type != "DENOISED":

            # Noise for P and S
            noise_psd = fitter.get_noise_spectrum_SP(component)[1:]
            noise_freq = fitter.get_noise_frequency_SP()[1:]

            sns.lineplot(
                ax=axP, x=noise_freq, y=noise_psd, label='noise (HF)', 
                color=colors["noise_high_sps"])
            
            sns.lineplot(
                ax=axS, x=noise_freq, y=noise_psd, label='noise (HF)', 
                color=colors["noise_high_sps"])

        # P
        P_psd = fitter.get_P_spectrum_SP(component)[1:]
        P_freq = fitter.get_P_frequency_SP()[1:]

        sns.lineplot(
            ax=axP, x=P_freq, y=P_psd, label='P phase (HF)', 
            color=colors["phase_p_high_sps"])

        # S
        S_psd = fitter.get_S_spectrum_SP(component)[1:]
        S_freq = fitter.get_S_frequency_SP()[1:]

        sns.lineplot(
            ax=axS, x=S_freq, y=S_psd, label='S phase (HF)', 
            color=colors["phase_s_high_sps"])

    # frequency ranges - P
    fmin = fitting_parameters.get_value(component, f'fminP')
    fmax = fitting_parameters.get_value(component, f'fmaxP')
    
    ## grey boxes
    
    if fmin and fmax:
        axP.axvspan(
            xmin=axP.get_xlim()[0], xmax=fmin, 
            facecolor=COLOR_SPECTRA_BOTTOM_FREQ_BOX, alpha=0.3)
        axP.axvspan(
            xmin=fmax, xmax=axP.get_xlim()[1], 
            facecolor=COLOR_SPECTRA_BOTTOM_FREQ_BOX, alpha=0.3)

    # frequency ranges - S
    fmin = fitting_parameters.get_value(component, f'fminS')
    fmax = fitting_parameters.get_value(component, f'fmaxS')
    
    if fmin and fmax:
        axS.axvspan(
            xmin=axS.get_xlim()[0], xmax=fmin, 
            facecolor=COLOR_SPECTRA_BOTTOM_FREQ_BOX, alpha=0.3)
        axS.axvspan(
            xmin=fmax, xmax=axS.get_xlim()[1], 
            facecolor=COLOR_SPECTRA_BOTTOM_FREQ_BOX, alpha=0.3)

    if fitting_parameters.get_value(None, 'is_manually_reviewed'):
        
        # manual fit curve P
        noise_freq = results[component][f'f_phase_P'][1:]
        fit = results[component][f'y_lorentz_new_plot_P'][1:]

        sns.lineplot(
            ax=axP, x=noise_freq, y=fit, label='manual fit', 
            color=colors["manual_fit"])

        # P uncertainty
        fmin, ymin, fmax, ymax = fitter.get_uncertainty(component, 'P')
        if fmin is not None and fmax is not None:
            
            # fmin, fmax are identical, it doesn't matter which one we use
            axP.fill_between(fmin[1:], ymin[1:], ymax[1:], alpha=0.2)

        # manual fit curve S
        noise_freq = results[component][f'f_phase_S'][1:]
        fit = results[component][f'y_lorentz_new_plot_S'][1:]

        sns.lineplot(
            ax=axS, x=noise_freq, y=fit, label='manual fit', 
            color=colors["manual_fit"])

        # S uncertainty
        fmin, ymin, fmax, ymax = fitter.get_uncertainty(component, 'S')
        if fmin is not None and fmax is not None:
            
            # fmin, fmax are identical, it doesn't matter which one we use
            axS.fill_between(fmin[1:], ymin[1:], ymax[1:], alpha=0.2)
    
    else:
        
        # automatic best fit curve P
        noise_freq = results[component][f'f_noise_maskedP']
        fit = results[component][f'y_best_fit_P']

        sns.lineplot(
            ax=axP, x=noise_freq, y=fit, label='best fit', 
            color=colors["phase_p"])
        
        # automatic best fit curve S
        noise_freq = results[component][f'f_noise_maskedS']
        fit = results[component][f'y_best_fit_S']

        sns.lineplot(
            ax=axS, x=noise_freq, y=fit, label='best fit', 
            color=colors["phase_s"])

    # A0
    _add_a0_marker(axP, axS, fitting_parameters, component)

    # fc
    _add_fc_marker(axP, axS, fitting_parameters, component)
    
    # axis limits
    if wf_type == "DENOISED":
        axP = _set_axis_limits_psd_denoised(axP, P_psd)
        axS = _set_axis_limits_psd_denoised(axS, S_psd)
    
    # automatic legends
    _ = axP.legend(loc='lower left')
    _ = axS.legend(loc='lower left')


def _add_origin_time_marker(axis, ot_plot):
    
    axis.axvline(x=ot_plot, color=COLOR_SPECTRA_TOP_OT, alpha=0.3)
    
    yaxis_min = axis.get_ylim()[0]
    ot_label_ycoord = yaxis_min + 0.05 * np.abs(yaxis_min)
    
    axis.annotate('OT', xy=(ot_plot, ot_label_ycoord), 
        color=COLOR_SPECTRA_TOP_OT, alpha=0.5)


def _add_a0_marker(axP, axS, fitting_parameters, component):
    
    axP.axhline(y=fitting_parameters.get_value(component, 'A0'), 
            color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.3)
    axP.axhline(y=fitting_parameters.get_value(component, 'A0-low'), 
                color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.3, linestyle='dashed')
    axP.axhline(y=fitting_parameters.get_value(component, 'A0-high'), 
                color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.3, linestyle='dashed')
    axP.annotate('A0',
            xy=(axP.get_xlim()[0], 
                fitting_parameters.get_value(component, 'A0')),
            color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.5)

    axS.axhline(y=fitting_parameters.get_value(component, 'A0'), 
            color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.3)
    axS.axhline(y=fitting_parameters.get_value(component, 'A0-low'), 
                color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.3, linestyle='dashed')
    axS.axhline(y=fitting_parameters.get_value(component, 'A0-high'), 
                color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.3, linestyle='dashed')
    axS.annotate('A0',
            xy=(axS.get_xlim()[0], 
                fitting_parameters.get_value(component, 'A0')),
            color=COLOR_SPECTRA_BOTTOM_A0, alpha=0.5)


def _add_fc_marker(axP, axS, fitting_parameters, component):
    
    axP.axvline(x=fitting_parameters.get_value(component, 'cornerfrequency'),
        color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.3)
    axP.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-low'), 
        color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.3, linestyle='dashed')
    axP.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-high'),
        color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.3, linestyle='dashed')
    axP.annotate(
        'fc', xy=(fitting_parameters.get_value(component, 'cornerfrequency'),
        axP.get_ylim()[0] + 1), color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.5)

    axS.axvline(x=fitting_parameters.get_value(component, 'cornerfrequency'),
        color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.3)
    axS.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-low'), 
        color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.3, linestyle='dashed')
    axS.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-high'),
        color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.3, linestyle='dashed')
    axS.annotate(
        'fc', xy=(fitting_parameters.get_value(component, 'cornerfrequency'),
        axS.get_ylim()[0] + 1), color=COLOR_SPECTRA_BOTTOM_F_CENTER, alpha=0.5)


def plot_filterbanks(
    catalog, 
    dir_out: str='filterbanks', 
    annotations: Annotations=None,
    wf_type: str="RAW", 
    normtype: str='none', 
    rotate: bool=False, 
    smprate: str ="", 
    orientation: list=[], 
    norm: list=[], 
    force_products: bool=False, 
    calculate_filterbanks: bool=False, 
    plot_filterbanks: bool=True):

    # print("catalog: available normtypes: {}".format(norm))
    # print("catalog: requested normtypes: {}".format(normtype))
    
    print("products: calculate/plot filterbanks for {} waveforms (smprate {}, "\
        "ZRT {})".format(wf_type, smprate, rotate))
    
    if normtype not in norm:
        print("catalog: plot_filterbanks, norm {} not requested".format(
            normtype))
        return
    
    if (not rotate and 'ZNE' not in orientation):
        print("catalog: plot_filterbanks, ZNE not requested")
        return 
    
    if (rotate and 'ZRT' not in orientation):
        print("catalog: plot_filterbanks, ZRT not requested")
        return 
    
    for event in tqdm(catalog, file=stdout):

        if event.waveforms_VBB is None:
            print("plot_filterbanks: event {}, no VBB waveforms exist, "\
                "skipping".format(event.name))
            continue
    
        if rotate and event.baz is None:
            print("plot_filterbanks: event {}, rotation to ZRT requested but "\
                "no BAZ exists, skipping".format(event.name))
            continue

        if wf_type != event.wf_type:
            raise ValueError("plot_filterbanks: event {}, wf_type {} was "\
                "requested, does not match event wf_type {}".format(
                    event.name, wf_type, event.wf_type))
        
        # set frequency metadata
        fmax_LF = 8.0
        fmin_LF = 1.0 / 32.0
        df_HF = 2.0**0.25
        
        fmax_HF = 16.0
        fmin_HF = 1.0 / 2.0
        df_LF = 2.0**0.5
        
        avail_rate = event.available_sampling_rates()
        
        if smprate == 'VBB_LF':
            if avail_rate['VBB_Z'] is None or \
                avail_rate['VBB_N'] is None or \
                avail_rate['VBB_E'] is None:
                continue
            instrument = 'VBB'
            fmin = fmin_LF
            fmax = fmax_LF
            df = df_LF
        
        elif smprate == 'SP_HF':
            if avail_rate['SP_Z'] != 100. or \
                avail_rate['SP_N'] != 100. or \
                avail_rate['SP_E'] != 100.:
                continue
            instrument = 'SP'
            fmin = fmin_HF
            fmax = fmax_HF
            df = df_HF
        
        elif smprate == 'LF+HF':
            if avail_rate['VBB100_Z'] == 100. and \
                avail_rate['VBB100_N'] == 100. and \
                avail_rate['VBB100_E'] == 100.:
                instrument = 'VBB+VBB100'
            
            elif avail_rate['SP_Z'] == 100. and \
                    avail_rate['SP_N'] == 100. and \
                    avail_rate['SP_E'] == 100.:
                instrument = 'VBB+SP'
            
            else:
                continue
            
            fmin = fmin_LF
            fmax = fmax_HF
            df = df_HF
        
        else:
            raise ValueError(f'Invalid value for smprate: {smprate}')

        # print("ev {}: plotting filterbanks for smprate {}, instrument "\
        #     "{}".format(event.name, smprate, instrument))
    
        # set pick metadata (only needed for 'phases' zoom level)
        if event.mars_event_type_short in ['LF', 'WB', 'BB']:
            
            # LF family
            if 'S' in event.picks and 'P' in event.picks and \
                    len(event.picks['S']) * len(event.picks['P']) > 0:
                t_S = utct(event.picks['S'])
                t_P = utct(event.picks['P'])
            else:
                t_P = utct(event.starttime)
                t_S = None
        
        elif event.mars_event_type_short in ['HF', '24', 'VF']:
            
            # HF family
            
            # NOTE: Pg and Sg have been retired
            if 'S' in event.picks and 'P' in event.picks and \
                    len(event.picks['S']) * len(event.picks['P']) > 0:
                t_S = utct(event.picks['S'])
                t_P = utct(event.picks['P'])
                
            else:
                t_P = utct(event.starttime)
                t_S = None
        
        else: 
            # Super High Frequency
            t_P = utct(event.starttime)
            t_S = None

        print("plot_filterbanks: filter traces for event {}, {}/Q{}, wf {}, "\
            "smprate {}, ZRT {}, instrument {}".format(event.name, 
                event.mars_event_type_short, event.quality, event.wf_type, 
                smprate, rotate, instrument))
            
        try:
            
            # log, waveforms, normwindow (annotations)
            # tmin_plot, tmax_plot, timemarkers, starttime, endtime (instrument)
            # f_VBB_SP_transition, fnam, station, location_code (wf_type)
            filter_traces_for_filterbanks(
                event, 
                fmin, 
                fmax, 
                df, 
                annotations=annotations, 
                instrument=instrument, 
                wf_type=wf_type, 
                normtype=normtype, 
                rotate=rotate, 
                smprate=smprate, 
                force_products=force_products, 
                calculate_filterbanks=calculate_filterbanks, 
                keep_filterbanks=plot_filterbanks)
            
            
        except Exception as e:
            print(f"Error products.filter_traces with event {event.name}: {e}")
            continue

        if not plot_filterbanks:
            continue
        
        
        ## FROM HERE ON IT IS PLOTTING
        
        # TODO(fab): move into plot function
        ev_folder = pjoin(dir_out, event.name)

        if not os.path.exists(ev_folder):
            os.makedirs(ev_folder)

        def plot_filename(ev, zoom):
            rot = 'ZRT' if rotate else 'ZNE'
            return pjoin(
                ev_folder,
                "filterbank_{}_Zoom_{}_SampRate_{}_Norm_{}_Rotation_{}_"\
                "Data_{}.png".format(
                    ev.name, zoom, smprate, normtype, rot, ev.wf_type))

        fnam = plot_filename(event, 'out')

        hasdata = False
        
        if not pexists(fnam) or force_products:
            
            try:
                print("catalog: plot filterbanks for event {}, {}/Q{}, "\
                    "wf {}, smprate {}, ZRT {}, norm {}".format(
                    event.name, event.mars_event_type_short, event.quality, 
                    event.wf_type, smprate, rotate, normtype))
        
                event.plot_filterbank(
                    normwindow='all', 
                    annotations=annotations,
                    starttime=event.starttime - 300.0,
                    endtime=event.endtime + 300.0,
                    instrument=instrument,
                    fnam=fnam, 
                    fmin=fmin, 
                    fmax=fmax, 
                    df=df,
                    normtype=normtype, 
                    rotate=rotate)
            
            except (AttributeError, IndexError) as err:
                print( f"Exception in filterbank for event "\
                    "{event.name}: {err}")
            
            else:
                hasdata = True

        if event.quality in ['A', 'B', 'C'] and hasdata:
            
            fnam = plot_filename(event, 'in')
            try:
                if not pexists(fnam) or force_products:
                    
                    # TODO(fab): use event.plot_parameters['filterbanks']['t_P']
                    print("catalog: plot filterbanks for event {}, {}/Q{}, "\
                        "wf {}, smprate {}, ZRT {}, norm {}".format(
                        event.name, event.mars_event_type_short, event.quality, 
                        event.wf_type, smprate, rotate, normtype))
                
                    event.plot_filterbank(
                        starttime=t_P - 300.,
                        endtime=t_P + 1100.,
                        normwindow='S',
                        annotations=annotations,
                        tmin_plot=-240., 
                        tmax_plot=900.,
                        fnam=fnam,
                        instrument=instrument,
                        fmin=fmin, 
                        fmax=fmax, 
                        df=df,
                        normtype=normtype, 
                        rotate=rotate)

                if t_S is not None:
                    fnam = plot_filename(event, 'phases')
                    if not pexists(fnam) or force_products:
                        print("catalog: plot filterbanks for event {}, {}/Q{}, "\
                            "wf {}, smprate {}, ZRT {}, norm {}".format(
                            event.name, event.mars_event_type_short, event.quality, 
                            event.wf_type, smprate, rotate, normtype))
                
                        event.plot_filterbank(
                            starttime=t_P - 120.,
                            endtime=t_S + 240.,
                            normwindow='S',
                            annotations=annotations,
                            tmin_plot=-50.,
                            tmax_plot=t_S - t_P + 200.,
                            fnam=fnam,
                            instrument=instrument,
                            fmin=fmin, 
                            fmax=fmax, 
                            df=df,
                            normtype=normtype, 
                            rotate=rotate)
            
            except (IndexError, AttributeError) as err:
                print(f"Exception in filterbank for event "\
                    "{event.name}: {err}")
                
        plt.close()


def filter_traces_for_filterbanks(
    event, 
    fmin: float=1.0/64.0, 
    fmax: float=4.0, 
    df: float=2**0.5,
    log: bool=False, 
    waveforms: bool=False, 
    normwindow: str='all',
    annotations: Annotations=None, 
    tmin_plot: float=None,
    tmax_plot: float=None, 
    timemarkers: dict=None,
    starttime: UTCDateTime=None, 
    endtime: UTCDateTime=None,
    instrument: str="", 
    f_VBB_SP_transition=7.5, 
    fnam: str=None,
    station: str=constants.DEFAULT_STATION_NAME,
    location_code: str=constants.DEFAULT_LOCATION_CODE, 
    wf_type="RAW", 
    normtype="none", 
    rotate=False, 
    smprate ="", 
    force_products: bool=False, 
    calculate_filterbanks: bool=False, 
    keep_filterbanks: bool=True):

    """
    
    """

    # from spectra, to be updated
    if not event._waveforms_read:
        raise RuntimeError('waveforms not read in Event object\n' +
                            'Call Event.read_waveforms() first.')

    event_path = pjoin("./events", "{}".format(event.name))
    filterbanks_path = pjoin(event_path, 'filterbanks')
    
    # event.wf_type, smprate, orientation, normtype
    orientation = "ZRT" if rotate else "ZNE" 
    
    # zoom parameters not needed for pickle
    # NOTE: normtype can be "none" which appears in pickle file name
    
    filterbanks_dict_path = pjoin(
        filterbanks_path, "{}_filterbanks_{}_{}_{}_{}.{}".format(
            event.name, smprate, normtype, orientation, wf_type,
            PICKLE_EXTENSION))
    
    # check if pickled spectra exist
    if not(calculate_filterbanks) and keep_filterbanks and pexists(
            filterbanks_dict_path):
        
        print("filter_traces: reading filterbanks from {}".format(
            filterbanks_dict_path))
        
        with io.open(filterbanks_dict_path, 'rb') as fin:
            filterbanks_dict = pickle.load(fin)
            
        event.filterbank_data = copy.deepcopy(filterbanks_dict['filterbanks'])
        
        del filterbanks_dict
        
        if len(event.filterbank_data) == 0:
            raise RuntimeError(
                "filter_traces: filterbanks dict has zero length")
        
        # print("calc_spectra: read spectra dict from pickle w/ keys: "\
        #     "{}".format(self.spectra.keys()))
        
        event._filterbanks_available = True
        
        return True
    
    event.filterbank_data = dict()
    
    # Determine frequencies
    event.filterbank_data['fmin'] = fmin
    event.filterbank_data['fmax'] = fmax
    event.filterbank_data['df'] = df
    
    nfreqs = int(np.round(np.log(fmax / fmin) / np.log(df), decimals=0) + 1)
    freqs = np.geomspace(fmin, fmax + 0.001, nfreqs)
    
    event.filterbank_data['nfreqs'] = nfreqs
    event.filterbank_data['freqs'] = freqs
    
    # print("nfreqs: {}, min freq: {}, max freq: {}".format(
    #     nfreqs, freqs[0], freqs[-1]))
    
    # print("waveforms VBB;\n{}".format(event.waveforms_VBB))
    # print("waveforms SP:\n{}".format(event.waveforms_SP))
    
    # Reference time
    if 'P' in event.picks and len(event.picks['P']) > 0 and \
        event.picks_methodid['P'] != PICK_METHOD_ALIGNED:
            
        t_ref = utct(event.picks['P'])
        t_ref_type = 'P'
    
    elif 'PP' in event.picks and len(event.picks['PP']) > 0 and \
        event.picks_methodid['PP'] != PICK_METHOD_ALIGNED:
            
        t_ref = utct(event.picks['PP'])
        t_ref_type = 'PP'
    
    else:
        t_ref = event.starttime
        t_ref_type = 'start time'
    
    event.filterbank_data['t_ref'] = t_ref
    event.filterbank_data['t_ref_type'] = t_ref_type
    
    # select from waveforms
    if instrument == 'VBB':
        st_LF = event.waveforms_VBB.select(channel='??[ENZ]').copy()
        st_HF = event.waveforms_VBB.select(channel='??[ENZ]').copy()

        st_LF_desc = f'LF: {st_LF[0].stats.station}.{st_LF[0].stats.location}.{st_LF[0].stats.channel[0:2]}@{st_LF[0].stats.sampling_rate}'
        st_HF_desc = ''
    
    elif instrument == 'VBB100':
        st_LF = event.waveforms_VBB100.select(channel='??[ENZ]').copy()
        st_HF = event.waveforms_VBB100.select(channel='??[ENZ]').copy()
        
        st_LF_desc = ''
        st_HF_desc = f'HF: {st_HF[0].stats.station}.{st_HF[0].stats.location}.{st_HF[0].stats.channel[0:2]}@{st_HF[0].stats.sampling_rate}'
    
    elif instrument == 'SP':
        st_LF = event.waveforms_SP.select(channel='??[ENZ]').copy()
        st_HF = event.waveforms_SP.select(channel='??[ENZ]').copy()
        st_LF_desc = ''
        st_HF_desc = f'HF: {st_HF[0].stats.station}.{st_HF[0].stats.location}.{st_HF[0].stats.channel[0:2]}@{st_HF[0].stats.sampling_rate}'
    
    elif instrument == 'VBB+VBB100':
        st_LF = event.waveforms_VBB.select(channel='??[ENZ]').copy()
        st_HF = event.waveforms_VBB100.select(channel='??[ENZ]').copy()
        
        st_LF_desc = f'LF: {st_LF[0].stats.station}.{st_LF[0].stats.location}.{st_LF[0].stats.channel[0:2]}@{st_LF[0].stats.sampling_rate}'
        st_HF_desc = f'HF: {st_HF[0].stats.station}.{st_HF[0].stats.location}.{st_HF[0].stats.channel[0:2]}@{st_HF[0].stats.sampling_rate}'
    
    elif instrument == 'VBB+SP':
        st_LF = event.waveforms_VBB.select(channel='??[ENZ]').copy()
        st_HF = event.waveforms_SP.select(channel='??[ENZ]').copy()
        
        st_LF_desc = f'LF: {st_LF[0].stats.station}.{st_LF[0].stats.location}.{st_LF[0].stats.channel[0:2]}@{st_LF[0].stats.sampling_rate}'
        st_HF_desc = f'HF: {st_HF[0].stats.station}.{st_HF[0].stats.location}.{st_HF[0].stats.channel[0:2]}@{st_HF[0].stats.sampling_rate}'
    
    else:
        raise ValueError(f'Invalid value for instrument: {instrument}')

    if rotate:
        st_HF.rotate('NE->RT', back_azimuth=event.baz)
        st_LF.rotate('NE->RT', back_azimuth=event.baz)

    event.filterbank_data['instrument'] = instrument
    event.filterbank_data['st_LF_desc'] = st_LF_desc
    event.filterbank_data['st_HF_desc'] = st_HF_desc
        
    tstart_norm = dict(
        P=event.picks.get('P_spectral_start', None), 
        S=event.picks.get('S_spectral_start', None), all=event.starttime)
    
    tend_norm = dict(
        P=event.picks.get('P_spectral_end', None),
        S=event.picks.get('S_spectral_end', None), all=event.endtime)
    
    # check tstart norm_for existence of requested phase
    # (same existence for tend_norm)
    # set normwindow in order 'S', 'P', 'all'
    if normwindow == 'S':
        try:
            tstart_norm = utct(tstart_norm[normwindow])
            tend_norm = utct(tend_norm[normwindow])
        except Exception:
            normwindow = 'P'
            
    if normwindow == 'P':
        try:
            tstart_norm = utct(tstart_norm[normwindow])
            tend_norm = utct(tend_norm[normwindow])
        except Exception:
            normwindow = 'all'
    
    # fallback 'all' is always there
    if normwindow == 'all':
        tstart_norm = utct(tstart_norm[normwindow])
        tend_norm = utct(tend_norm[normwindow])

    event.filterbank_data['normwindow'] = normwindow
    
    event.filterbank_data['tstart_norm'] = tstart_norm
    event.filterbank_data['tend_norm'] = tend_norm
    
    if starttime is None:
        starttime = event.starttime - 100.0
    if endtime is None:
        endtime = event.endtime + 100.0
        
    if tmin_plot is None:
        tmin_plot = starttime - t_ref
    if tmax_plot is None:
        tmax_plot = endtime - t_ref
    
    event.filterbank_data['starttime'] = starttime
    event.filterbank_data['endtime'] = endtime
    
    event.filterbank_data['tmin_plot'] = tmin_plot
    event.filterbank_data['tmax_plot'] = tmax_plot
    
    # print("t_ref: {}, starttime: {}, endtime: {}".format(
    #     t_ref, starttime, endtime))
    
    for st in (st_HF, st_LF):
        st.trim(
            starttime=utct(starttime) - 1.0 / fmin,
            endtime=utct(endtime) + 1.0 / fmin)

    event.filterbank_data['st_LF'] = st_LF
    event.filterbank_data['st_HF'] = st_HF
    
    # compute normalization factors
    maxfac_all = None
    offset_all = None
    maxfac_tr = {}
    offset_tr = {}
    
    trids = ('Z','2','3')
    
    for trid in trids:
        maxfac_tr[trid] = None
        offset_tr[trid] = None

    freqs_data = {}

    xvec_env = []
    xvec = []
    
    # 1st loop over frequencies to filter, rotate, and get norm factors
    for ifreq, fcenter in enumerate(freqs):

        f0 = fcenter / df
        f1 = fcenter * df

        # skip_freq_bin = False
        
        if fcenter < f_VBB_SP_transition:
            st_filt = st_LF.copy()
        else:
            st_filt = st_HF.copy()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                # print("filtering freq bin {}: center {}, min {}, max "\
                #     "{}".format(ifreq, fcenter, f0, f1))
                
                st_filt.filter('bandpass', freqmin=f0, freqmax=f1,
                                corners=FILTERBANK_CORNERS_COUNT)
        
        # f0 is above Nyquist
        except ValueError:  
            print("ev {}: Nyquist error: no 20sps data available for "\
                "event".format(event.name))
            continue

        st_filt.trim(starttime=utct(starttime),  endtime=utct(endtime))

        if not st_filt:
            #print('No data available for event %s' % event.name)
            continue

        if rotate:
            tr_3 = st_filt.select(channel='??T')[0]
            tr_2 = st_filt.select(channel='??R')[0]
        else:
            tr_2 = st_filt.select(channel='??N')[0]
            tr_3 = st_filt.select(channel='??E')[0]

        tr_Z = st_filt.select(channel='??Z')[0]

        tr_2_env = envelope_smooth(tr=tr_2, mode='same',
                                    envelope_window_in_sec=10.)
        tr_3_env = envelope_smooth(tr=tr_3, mode='same',
                                    envelope_window_in_sec=10.)
        tr_Z_env = envelope_smooth(tr=tr_Z, mode='same',
                                    envelope_window_in_sec=10.)

        freqs_data[ifreq] = {}
        freqs_data[ifreq]['fcenter'] = fcenter

        freqs_data[ifreq]['tr'] = {}
        freqs_data[ifreq]['tr']['Z'] = tr_Z
        freqs_data[ifreq]['tr']['2'] = tr_2
        freqs_data[ifreq]['tr']['3'] = tr_3

        freqs_data[ifreq]['tr_env'] = {}
        freqs_data[ifreq]['tr_env']['Z']= tr_Z_env
        freqs_data[ifreq]['tr_env']['2']= tr_2_env
        freqs_data[ifreq]['tr_env']['3']= tr_3_env

        freqs_data[ifreq]['maxfac'] = {}
        freqs_data[ifreq]['offset'] = {}
        
        for trid, tr in zip(trids, (tr_Z_env, tr_2_env, tr_3_env) ):

            if log:
                tr_norm = tr.slice(starttime=tstart_norm,
                                    endtime=tend_norm)
                maxfac = np.quantile(tr_norm.data, q=0.8)
                offset = np.quantile(tr_norm.data, q=0.1)
            
            else:
                tr_norm = tr.slice(starttime=tstart_norm,
                                    endtime=tend_norm,
                                    nearest_sample=True)
                try:
                    maxfac = np.quantile(tr_norm.data, q=0.8)
                    offset = np.quantile(tr_norm.data, q=0.1)
                except:
                    maxfac = 1.e-9
                    offset = 0.

            freqs_data[ifreq]['maxfac'][trid] = maxfac
            freqs_data[ifreq]['offset'][trid] = offset

            if maxfac_all is None or maxfac_all < maxfac:
                maxfac_all = maxfac
            if offset_all is None or offset_all > offset:
                offset_all = offset

            if maxfac_tr[trid] is None or maxfac_tr[trid] < maxfac:
                maxfac_tr[trid] = maxfac
            if offset_tr[trid] is None or offset_tr[trid] > offset:
                offset_tr[trid] = offset

    # print("2nd freq loop")
    # 2nd loop over frequencies to plot traces
    for ifreq, fcenter in enumerate(freqs):

        if ifreq not in freqs_data:
            continue

        if fcenter != freqs_data[ifreq]['fcenter']:
            raise RuntimeError(
                'Internal logic error while bulding filterbanks')

        # print("freq bin {}: center {} compute offsets".format(
        #     ifreq, fcenter))
        
        tr_Z     = freqs_data[ifreq]['tr']['Z']
        tr_Z_env = freqs_data[ifreq]['tr_env']['Z']

        t_offset = float(tr_Z_env.stats.starttime - t_ref)
        xvec_env = tr_Z_env.times() + t_offset
        xvec = tr_Z.times() + t_offset

        # three orientation subplots
        for itr, trid in enumerate(trids):

            maxfac = None
            offset = None
            
            if normtype == 'none':
                maxfac = freqs_data[ifreq]['maxfac'][trid]
                offset = freqs_data[ifreq]['offset'][trid]
            
            elif normtype == 'single_components':
                maxfac = maxfac_tr[trid]
                offset = offset_tr[trid]
            
            elif normtype == 'all_components':
                maxfac = maxfac_all
                offset = offset_all
            
            else:
                raise ValueError(f'Invalid value for normtype: {normtype}')

            tr       = freqs_data[ifreq]['tr'][trid]
            tr_env   = freqs_data[ifreq]['tr_env'][trid]

            # ax[itr].plot(xvec_env,
            #              iangle + tr_Z_env.data / maxfac, c='grey',
            #              lw=1)
            # ax[itr].fill_between(x=xvec_env,
            #                      y1=iangle + tr_Z_env.data / maxfac,
            #                      y2=iangle, color='darkgrey')

            # if log:
            #     ax[itr].plot(xvec_env,
            #                     ifreq + np.log(tr_env.data / maxfac) / 3,
            #                     lw=1.0, zorder=50)
            # else:
            # 
            #     if waveforms:
            #         color = 'k'
            #     else:
            #         color = 'C%d' % (ifreq % 10)
            # 
            # 
            #     ax[itr].plot(xvec_env,
            #                     ifreq + (tr_env.data - offset) / maxfac,
            #                     c=color,
            #                     lw=0.5, zorder=80)
            #     if waveforms:
            #         ax[itr].plot(xvec,
            #                         ifreq + tr.data / maxfac,
            #                         c='C%d' % (ifreq % 10),
            #                         lw=0.5, zorder=50 - ifreq)

                    # plot envelopes
#                     ax[itr].plot(
#                         xvec_env[ifreq],
#                         ifreq + \
#                             (envelopes[itr][ifreq].data - offset) / max_maxfac,
#                         c=color, lw=0.5, zorder=80)
#                     
#                     if waveforms:
#                         # plot waveforms
#                         ax[itr].plot(
#                             xvec[ifreq],
#                             ifreq + waveform_tr[itr][ifreq].data / max_maxfac,
#                             c='C%d' % (ifreq % 10),
#                             lw=0.5, zorder=50 - ifreq)

    event.filterbank_data['freqs_data'] = freqs_data

    event.filterbank_data['xvec_env'] = xvec_env
    event.filterbank_data['xvec'] = xvec
    
    event._filterbanks_available = True

    # write filterbanks to pickle files, in event/Sxxxxy/ directory
    if calculate_filterbanks:
        print("filter_traces: writing filterbanks to {}".format(
            filterbanks_dict_path))
        
        makedirs(filterbanks_path, exist_ok=True)
        
        filterbanks_dict = {}
        filterbanks_dict['filterbanks'] = copy.deepcopy(event.filterbank_data)
        
        with io.open(filterbanks_dict_path, 'wb') as of:
            pickle.dump(filterbanks_dict, of, pickle.HIGHEST_PROTOCOL)
                        
        del filterbanks_dict
    
    if not keep_filterbanks:
        print("filter_traces: deleting filterbanks from memory")
        event.filterbank_data = None
