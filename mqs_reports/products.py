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
import os.path
import sys

from os.path import exists as pexists, join as pjoin

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from obspy import UTCDateTime

from tqdm import tqdm

from fittingparam import FittingParameterPool

from mqs_reports.utils import add_orientation_to_stream_info

sns.set_theme(style="darkgrid")


def plot_spectra(fitter,
                 fitting_parameters,
                 fitting_parameters_defaults,
                 dir_out: str,
                 winlen_sec: float,
                 wf_type : str,
                 rotate: bool=False,
                 smprate: str="",
                 orientation: list=[],
                 force_products: bool=False) -> None:
    
    print("fitter: calculate/plot spectra for {} waveforms (smprate {}, "\
        "ZRT {})".format(wf_type, smprate, rotate))
    
    for event in tqdm(fitter.catalog, file=sys.stdout):
        
        if event.waveforms_VBB is None:
            print("fitter: event {}, no VBB waveforms exist, skipping".format(
                event.name))
            continue
        
        if rotate and event.baz is None:
            print("fitter: event {}, rotation to ZRT requested but no BAZ "\
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

        print("fitter: swap events for event {}, {}/Q{}, wf {}, smprate "\
            "{}, ZRT {}".format(event.name, event.mars_event_type_short, 
                event.quality, event.wf_type, smprate, rotate))
            
        try:
            fitter.swap_event(
                event_name=event.name,
                detick_nfsamp=(10 if wf_type != "DEGLITCHED" else 0),
                instrument=instrument, rotate=rotate,
                time_windows=spectral_windows, smprate=smprate, 
                force_products=force_products)
            
        except Exception as e:
            print(f"Error fitter.swap_event with event {event.name}: {e}")
            continue

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

        #
        # add missing info for component R and T
        #
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
        
        for component in (['R','T'] if rotate else ['Z','N','E']):

            tr = stream.select(channel='*'+component)[0].copy()

            fnam = plot_filename(fitter.event, component)

            if pexists(fnam) and not(force_products):
                print("fitter.plot_spectra: plot file {} exists, "\
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

            print("fitter.plot_spectra: create figure for plot file {}".format(
                fnam))
            
            fig = plt.figure(figsize=(20,12))
            fig.subplots_adjust(top=0.911,  bottom=0.097,
                                left=0.049, right=0.972,
                                hspace=0.2, wspace=0.116)
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])

            # adjusted plot ttile
            # fig.suptitle(
            #     f'Event={fitter.event.name} LQ={fitter.event.quality} "\
            #     "Type={fitter.event.mars_event_type_short} "\
            #     "Component={component} {LF_streaminfo} {HF_streaminfo}')
            
            if len(LF_streaminfo) > 0:
                LF_streaminfo_with_orientation = add_orientation_to_stream_info(
                    LF_streaminfo, component)
            
            if len(HF_streaminfo) > 0:
                HF_streaminfo_with_orientation = add_orientation_to_stream_info(
                    HF_streaminfo, component)
            
            fig.suptitle("Event {} {}/Q{} {} {}".format(
                    fitter.event.name, fitter.event.mars_event_type_short, 
                    fitter.event.quality, LF_streaminfo_with_orientation, 
                    HF_streaminfo_with_orientation), fontsize='x-large')
            
            _plot_spectra_top(
                    fitter, ax1, tr, component, spectral_windows,
                    fitting_parameters_pool
            )
            _plot_spectra_bottom(
                    ax2, ax3, fitter, LF_streaminfo, HF_streaminfo, component,
                    fitting_parameters_pool, results, wf_type
            )
            
            fig.savefig(fnam)
            plt.close(fig)


def _plot_spectra_top(fitter, ax, tr, component, windows, fitting_parameters):

    #if parameters.filter_apply:
    #    tr.filter('bandpass', freqmin=parameters.filter_min_freq, 
    #              freqmax=parameters.filter_max_freq, 
    #              zerophase=parameters.filter_zero_phase,
    #              corners=parameters.filter_order)
    #    tr.taper(max_length=60, max_percentage=0.3)

    sns.lineplot(ax=ax, x=tr.times(), y=tr.data, color='steelblue')

    # this is stream_info with orientation
    ax.set(xlabel=f'{tr.id}@{tr.stats.sampling_rate}')

    to_tr_time = lambda time_str: UTCDateTime(time_str) - tr.stats.starttime

    data_min = np.min(tr.data)
    data_max = np.max(tr.data)

    noise_start = to_tr_time(windows['noise_start'])
    noise_end   = to_tr_time(windows['noise_end'])
    p_start     = to_tr_time(windows['P_spectral_start'])
    p_end       = to_tr_time(windows['P_spectral_end'])
    s_start     = to_tr_time(windows['S_spectral_start'])
    s_end       = to_tr_time(windows['S_spectral_end'])

    rect = patches.Rectangle(xy=(noise_start, data_min),
                             width=noise_end-noise_start,
                             height=data_max-data_min,
                             linewidth=3, edgecolor='darkgray',
                             alpha=0.5, facecolor="none")
    ax.add_patch(rect)

    rect = patches.Rectangle(xy=(p_start, data_min),
                             width=p_end-p_start,
                             height=data_max-data_min,
                             linewidth=3, edgecolor='red',
                             alpha=0.3, facecolor="none")
    ax.add_patch(rect)

    rect = patches.Rectangle(xy=(s_start, data_min),
                             width=s_end-s_start,
                             height=data_max-data_min,
                             linewidth=3, edgecolor='blue',
                             alpha=0.3, facecolor="none")
    ax.add_patch(rect)

    # Seismic phases
    try:
        if fitter.get_event_type() in ['LF', 'WB', 'BB']:
            p_phase = fitter.get_pick('P') or fitter.get_pick('PP') or \
                fitter.get_pick('P1') or fitter.get_pick('x1') or \
                    fitter.get_pick('Pg') or fitter.get_pick('y1') or \
                        fitter.get_pick('start')
            
            s_phase = fitter.get_pick('S') or fitter.get_pick('SS') or \
                fitter.get_pick('S1') or fitter.get_pick('x2') or \
                    fitter.get_pick('Sg') or fitter.get_pick('y2') or \
                        fitter.get_pick('start')
        else:
            p_phase = fitter.get_pick('P') or fitter.get_pick('PP') or \
                fitter.get_pick('P1') or fitter.get_pick('Pg') or \
                    fitter.get_pick('y1') or fitter.get_pick('x1') or \
                        fitter.get_pick('start')
            
            s_phase = fitter.get_pick('S') or fitter.get_pick('SS') or \
                fitter.get_pick('S1') or fitter.get_pick('Sg') or \
                    fitter.get_pick('y2') or fitter.get_pick('x2') or \
                        fitter.get_pick('start')
    except:
        p_phase = fitter.get_pick('P') or fitter.get_pick('P1') or \
            fitter.get_pick('x1') or fitter.get_pick('Pg') or \
                fitter.get_pick('y1') or fitter.get_pick('start')
            
        s_phase = fitter.get_pick('S') or fitter.get_pick('S1') or \
            fitter.get_pick('x2') or fitter.get_pick('Sg') or \
                fitter.get_pick('y2') or fitter.get_pick('start')

    # Mark the seismic phases
    width = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.005
    if p_phase:
        P = to_tr_time(p_phase)
        ax.axvspan(xmin=P-width/2, xmax=P+width/2, facecolor='red', alpha=0.3)
    if s_phase:
        S = to_tr_time(s_phase)
        ax.axvspan(xmin=S-width/2, xmax=S+width/2, facecolor='blue', alpha=0.3)


def _plot_spectra_bottom(axP, axS, fitter, LF, HF, component, fitting_parameters, results, wf_type):

    axP.set(xscale='log')
    axS.set(xscale='log')

    axP.set(xlabel='Frequency [Hz]', ylabel='Disp. PSD [db]')
    axS.set(xlabel='Frequency [Hz]', ylabel='Disp. PSD [db]')

    axP.set_title(f'P phase [{component}]')
    axS.set_title(f'S phase [{component}]')

    colors = sns.color_palette()

    #
    # VBB
    #
    if LF:

        # noise is too low amplitude and affects the y-axis range
        if wf_type != "DENOISED":

            # Noise for P and S
            noise_psd = fitter.get_noise_spectrum(component)[1:]
            noise_freq = fitter.get_noise_frequency()[1:]

            sns.lineplot(ax=axP, x=noise_freq, y=noise_psd, label='noise', color=colors[7])
            sns.lineplot(ax=axS, x=noise_freq, y=noise_psd, label='noise', color=colors[7])

        # P
        P_psd = fitter.get_P_spectrum(component)[1:]
        P_freq = fitter.get_P_frequency()[1:]

        sns.lineplot(ax=axP, x=P_freq, y=P_psd, label='phase', color=colors[1])

        # S
        S_psd = fitter.get_S_spectrum(component)[1:]
        S_freq = fitter.get_S_frequency()[1:]

        sns.lineplot(ax=axS, x=S_freq, y=S_psd, label='phase', color=colors[1])

    #
    # SP
    #
    if HF:

        # noise is too low amplitude and affects the y-axis range
        if wf_type != "DENOISED":

            # Noise for P and S
            noise_psd = fitter.get_noise_spectrum_SP(component)[1:]
            noise_freq = fitter.get_noise_frequency_SP()[1:]

            sns.lineplot(ax=axP, x=noise_freq, y=noise_psd, label='noise, high sps', color=colors[7])
            sns.lineplot(ax=axS, x=noise_freq, y=noise_psd, label='noise, high sps', color=colors[7])

        # P
        P_psd = fitter.get_P_spectrum_SP(component)[1:]
        P_freq = fitter.get_P_frequency_SP()[1:]

        sns.lineplot(ax=axP, x=P_freq, y=P_psd, label='phase, high sps', color=colors[9])

        # S
        S_psd = fitter.get_S_spectrum_SP(component)[1:]
        S_freq = fitter.get_S_frequency_SP()[1:]

        sns.lineplot(ax=axS, x=S_freq, y=S_psd, label='phase, high sps', color=colors[9])

    # frequency ranges - P
    fmin = fitting_parameters.get_value(component, f'fminP')
    fmax = fitting_parameters.get_value(component, f'fmaxP')
    if fmin and fmax:
        axP.axvspan(xmin=axP.get_xlim()[0], xmax=fmin, facecolor='darkgray', alpha=0.3)
        axP.axvspan(xmin=fmax, xmax=axP.get_xlim()[1], facecolor='darkgray', alpha=0.3)

    # frequency ranges - S
    fmin = fitting_parameters.get_value(component, f'fminS')
    fmax = fitting_parameters.get_value(component, f'fmaxS')
    if fmin and fmax:
        axS.axvspan(xmin=axS.get_xlim()[0], xmax=fmin, facecolor='darkgray', alpha=0.3)
        axS.axvspan(xmin=fmax, xmax=axS.get_xlim()[1], facecolor='darkgray', alpha=0.3)

    if fitting_parameters.get_value(None, 'is_manually_reviewed'):
        # manual fit curve P
        noise_freq = results[component][f'f_phase_P'][1:]
        fit = results[component][f'y_lorentz_new_plot_P'][1:]

        sns.lineplot(ax=axP, x=noise_freq, y=fit, label='manual fit', color=colors[2])

        # P uncertainty
        fmin, ymin, fmax, ymax = fitter.get_uncertainty(component, 'P')
        if fmin is not None and fmax is not None:
            # fmin, fmax are identical, it doesn't matter which one we use
            axP.fill_between(fmin[1:], ymin[1:], ymax[1:], alpha=0.2)

    else:
        # best fit curve P
        noise_freq = results[component][f'f_noise_maskedP']
        fit = results[component][f'y_best_fit_P']

        sns.lineplot(ax=axP, x=noise_freq, y=fit, label='best fit', color=colors[3])


    if fitting_parameters.get_value(None, 'is_manually_reviewed'):
        # manual fit curve S
        noise_freq = results[component][f'f_phase_S'][1:]
        fit = results[component][f'y_lorentz_new_plot_S'][1:]

        sns.lineplot(ax=axS, x=noise_freq, y=fit, label='manual fit', color=colors[2])

        # S uncertainty
        fmin, ymin, fmax, ymax = fitter.get_uncertainty(component, 'S')
        if fmin is not None and fmax is not None:
            # fmin, fmax are identical, it doesn't matter which one we use
            axS.fill_between(fmin[1:], ymin[1:], ymax[1:], alpha=0.2)

    else:
        # best fit curve S
        noise_freq = results[component][f'f_noise_maskedS']
        fit = results[component][f'y_best_fit_S']

        sns.lineplot(ax=axS, x=noise_freq, y=fit, label='best fit', color=colors[3])

    # A0
    axP.axhline(y=fitting_parameters.get_value(component, 'A0'), 
            color='cornflowerblue', alpha=0.3)
    axP.axhline(y=fitting_parameters.get_value(component, 'A0-low'), 
                color='cornflowerblue', alpha=0.3, linestyle='dashed')
    axP.axhline(y=fitting_parameters.get_value(component, 'A0-high'), 
                color='cornflowerblue', alpha=0.3, linestyle='dashed')
    axP.annotate('A0',
            xy=(axP.get_xlim()[0], 
                fitting_parameters.get_value(component, 'A0')),
            color='cornflowerblue', alpha=0.5)

    axS.axhline(y=fitting_parameters.get_value(component, 'A0'), 
            color='cornflowerblue', alpha=0.3)
    axS.axhline(y=fitting_parameters.get_value(component, 'A0-low'), 
                color='cornflowerblue', alpha=0.3, linestyle='dashed')
    axS.axhline(y=fitting_parameters.get_value(component, 'A0-high'), 
                color='cornflowerblue', alpha=0.3, linestyle='dashed')
    axS.annotate('A0',
            xy=(axS.get_xlim()[0], 
                fitting_parameters.get_value(component, 'A0')),
            color='cornflowerblue', alpha=0.5)

    # fc
    axP.axvline(x=fitting_parameters.get_value(component, 'cornerfrequency'),
            color='crimson', alpha=0.3)
    axP.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-low'), 
                color='crimson', alpha=0.3, linestyle='dashed')
    axP.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-high'),
                color='crimson', alpha=0.3, linestyle='dashed')
    axP.annotate('fc', 
            xy=(fitting_parameters.get_value(component, 'cornerfrequency'),
                axP.get_ylim()[0] + 1),
            color='crimson', alpha=0.5)

    axS.axvline(x=fitting_parameters.get_value(component, 'cornerfrequency'),
            color='crimson', alpha=0.3)
    axS.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-low'), 
                color='crimson', alpha=0.3, linestyle='dashed')
    axS.axvline(x=fitting_parameters.get_value(component, 'cornerfreq-high'),
                color='crimson', alpha=0.3, linestyle='dashed')
    axS.annotate('fc', 
            xy=(fitting_parameters.get_value(component, 'cornerfrequency'),
                axS.get_ylim()[0] + 1),
            color='crimson', alpha=0.5)

