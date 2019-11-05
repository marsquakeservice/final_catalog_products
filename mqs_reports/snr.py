#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

import numpy as np
from mqs_reports.event import Event


def calc_SNR(event: Event, fmin: float, fmax: float) -> float:
    p_noise = event.spectra['noise']['p_Z']
    df_noise = event.spectra['noise']['f'][1]
    f_bool = np.array((event.spectra['noise']['f'] > fmin,
                       event.spectra['noise']['f'] < fmax)).all(axis=0)
    power_noise = np.trapz(p_noise[f_bool], dx=df_noise)

    for type in ['S', 'P', 'all']:
        if type in event.spectra:
            p_signal = event.spectra[type]['p_Z']
            df_signal = event.spectra[type]['f'][1]
            f_bool = np.array((event.spectra[type]['f'] > fmin,
                               event.spectra[type]['f'] < fmax)).all(axis=0)
            continue

    power_signal = np.trapz(p_signal[f_bool], dx=df_signal)
    return power_signal / power_noise


def calc_stalta(event: Event,
                fmin: float, fmax: float,
                len_sta=100, len_lta=1000) -> float:
    from obspy.signal.trigger import classic_sta_lta
    import matplotlib.pyplot as plt
    tr_stalta = event.waveforms_VBB.select(channel='BHZ')[0].copy()
    tr_stalta.differentiate()
    tr_stalta.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=6,
                     zerophase=True)
    nsta = len_sta * tr_stalta.stats.sampling_rate
    nlta = len_lta * tr_stalta.stats.sampling_rate
    chf = classic_sta_lta(tr_stalta, nlta=nlta, nsta=nsta)
    tr_stalta.data = chf

    plt.plot(tr_stalta.times() + float(tr_stalta.stats.starttime),
             tr_stalta.data)
    tr_stalta.trim(starttime=event.starttime,
                   endtime=event.endtime)
    plt.plot(tr_stalta.times() + float(tr_stalta.stats.starttime),
             tr_stalta.data, 'r')
    plt.savefig('./tmp/stalta_%s.png' % event.name)
    plt.close('all')

    return np.max(tr_stalta.data)
