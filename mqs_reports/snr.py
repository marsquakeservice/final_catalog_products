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

from typing import Tuple

import numpy as np

from mqs_reports.event import Event
from typing import Tuple

def calc_SNR(event: Event, fmin: float, fmax: float,
             hor=False, SP=False) -> Tuple[float, str]:
    if SP:
        spectra = event.spectra_SP
    else:
        spectra = event.spectra
    if hor:
        comp = 'p_H'
    else:
        comp = 'p_Z'

    if 'noise' in spectra and  comp in spectra['noise']:
        p_noise = spectra['noise'][comp]
        df_noise = spectra['noise']['f'][1]
        f_bool = np.array((spectra['noise']['f'] > fmin,
                           spectra['noise']['f'] < fmax)).all(axis=0)
        power_noise = np.trapz(p_noise[f_bool], dx=df_noise)
        for spec_win in ['S', 'P', 'all']:
            if spec_win in spectra:
                p_signal = spectra[spec_win][comp]
                df_signal = spectra[spec_win]['f'][1]
                f_bool = np.array((spectra[spec_win]['f'] > fmin,
                                   spectra[spec_win]['f'] < fmax)).all(axis=0)
                break
        try:
            power_signal = np.trapz(p_signal[f_bool], dx=df_signal)
        except UnboundLocalError:
            power_signal = 0.0
            spec_win = 'none'

        return power_signal / power_noise, spec_win
    else:
        return -1, 'No Noise'


def calc_stalta(event: Event,
                fmin: float, fmax: float,
                len_sta=100, len_lta=1000) -> float:
    from obspy.signal.trigger import classic_sta_lta
    if event.waveforms_VBB is None:
        return 0.0
    else:
        tr_stalta = event.waveforms_VBB.select(channel='??Z')[0].copy()
        tr_stalta.differentiate()
        tr_stalta.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=6,
                         zerophase=True)
        nsta = len_sta * tr_stalta.stats.sampling_rate
        nlta = len_lta * tr_stalta.stats.sampling_rate
        try:
            chf = classic_sta_lta(tr_stalta, nlta=nlta, nsta=nsta)
        except Exception as e:
            print('Cannot compute stalta for event %s: %s' % (event.name,e))
            return 0.0
        tr_stalta.data = chf

        #plt.plot(tr_stalta.times() + float(tr_stalta.stats.starttime),
        #         tr_stalta.data)
        tr_stalta.trim(starttime=event.starttime,
                       endtime=event.endtime)
        #plt.plot(tr_stalta.times() + float(tr_stalta.stats.starttime),
        #         tr_stalta.data, 'r')
        #plt.savefig('./tmp/stalta_%s.png' % event.name)
        #plt.close('all')

        return np.max(tr_stalta.data)
