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

import glob

from os.path import join as pjoin

import matplotlib.pyplot as plt
from matplotlib import mlab as mlab
from matplotlib.patches import Rectangle

import numpy as np

import obspy
from obspy import UTCDateTime as utct

from obspy.signal.filter import envelope
from obspy.signal.rotate import rotate2zne
from obspy.signal.util import next_pow_2

from scipy.fftpack import fft, ifft
from scipy.signal import hilbert

import mqs_reports.constants as constants

from mqs_reports.constants import SEC_PER_DAY_EARTH, SEC_PER_DAY_MARS

from marsprocessingtools import utils as marsutils


def solify(UTC_time, sol0=constants.TIMESTAMP_SOL0):
    
    if type(UTC_time) is str:
        UTC_time = utct(UTC_time)
    
    MIT = (UTC_time - sol0) / SEC_PER_DAY_MARS
    t = utct((MIT - 1) * SEC_PER_DAY_EARTH)
    
    return t


def UTCify(LMST_time, sol0=constants.TIMESTAMP_SOL0):
    
    MIT = float(LMST_time) / SEC_PER_DAY_EARTH + 1
    UTC_time = utct(MIT * SEC_PER_DAY_MARS + float(sol0))
    
    return UTC_time


def create_fnam_event(timestamp, station, sc3dir, filenam_inst):
    
    dirnam = pjoin(
        sc3dir, 
        "op/data/waveform/{:04d}/XB/{}/".format(utct(timestamp).year, station))

    dirnam_inst = pjoin(dirnam, '???.D')
    hour = utct(timestamp).strftime('%H')
    
    fnam_inst = pjoin(
        dirnam_inst, 
        filenam_inst % (utct(timestamp).year, utct(timestamp).julday))
            
    if hour in ['00', '22', '23']:
        fnam_inst = fnam_inst[:-1] + '?'

    return fnam_inst


def f_c(M0, vs, ds):
    # Calculate corner frequency for event with M0,
    # assuming a stress drop ds
    return 4.9e-1 * vs * (ds / M0) ** (1 / 3)


def M0(Mw):
    return 10 ** (Mw * 1.5 + 9.1)


def attenuation_term(freqs, Qm, Qk=5e4, x=1e6, phase='S', vp=7.5e3, vs=4.1e3):
    if phase == 'P':
        L = 4 / 3 * (vs / vp) ** 2
        Q = 1 / (L / Qm + (1 - L) / Qk)
    else:
        Q = Qm
    return np.exp(-np.pi * x * freqs / vs / Q)


def pred_spec(freqs, ds, Qm, amp, dist, mag, phase, vs=5.e3):
    stf_amp = 1 / (1 + (freqs / f_c(M0=M0(mag),
                                    vs=vs, ds=ds)
                        ) ** 2)
    A = attenuation_term(freqs, Qm=Qm, x=dist, vs=vs, phase=phase)
    return 20 * np.log10(A * stf_amp) + amp


# def attenuation_term(freqs, Qm, Qk=5e4, x=1e6, phase='S', exp=0.0, f0=1,
#                      vp=7.5e3, vs=4.1e3):
#     if phase == 'P':
#         vel = vp
#         L = 4 / 3 * (vs / vp) ** 2
#         Q = 1 / (L / Qm + (1 - L) / Qk)
#     else:
#         vel = vs
#         Q = Qm
#     Q = Q * (freqs / f0) ** exp
#     Qscat = 300
#     Q = 1. / (1. / Q + 1 / Qscat)
#     return np.exp(-np.pi * x / vel * freqs / Q)
#
#
# def pred_spec(freqs, ds, Qm, amp, dist, mag, phase='S', vp=7.5e3, vs=4.2e3):
#     stf_amp = 1 / (1 + (freqs / f_c(M0=M0(mag),
#                                     vs=2.8e3, ds=ds)
#                         )                    filenam_pressure = 'XB.ELYSE.02.MDO.D.%d.%03d')
#     A = attenuation_term(freqs, Qm=Qm, x=dist, phase=phase, vp=vp, vs=vs)
#     return 20 * np.log10(A * stf_amp) + amp


def create_ZNE_HG(
    st: obspy.Stream,
    inv: obspy.Inventory=None,
    station=constants.DEFAULT_STATION_NAME,
    location_code=constants.DEFAULT_LOCATION_CODE):
    """
    Rotate sensor's original UVW directions into ZNE and ZRT.
    
    """
    
    if len(st) == 1 and st[0].stats.channel == 'EHU':
        
        # special treatment EH?
        # only SP1==SPZ switched on
        
        tr_Z = st[0].copy()
        tr_Z.stats.channel = 'EHZ'
        st_ZNE = obspy.Stream(traces=[tr_Z])

    else:
        
        chan_name = st[0].stats.channel[0:2]

        channelnames = dict(
            U="{}U".format(chan_name), 
            V="{}V".format(chan_name), 
            W="{}W".format(chan_name))
        
        if inv is None:
            
            sensor_dip = constants.SENSOR_DIRECTIONS['dip']
            sensor_azi = constants.SENSOR_DIRECTIONS['azimuth']
        
        else:

            chan_u = inv.select(
                station=st[0].stats.station, starttime=st[0].stats.starttime,
                endtime=st[0].stats.endtime, channel=channelnames['U'])[0][0][0]
            
            chan_v = inv.select(
                station=st[0].stats.station, starttime=st[0].stats.starttime,
                endtime=st[0].stats.endtime, channel=channelnames['V'])[0][0][0]
            
            chan_w = inv.select(
                station=st[0].stats.station, starttime=st[0].stats.starttime,
                endtime=st[0].stats.endtime, channel=channelnames['W'])[0][0][0]
            
            
            sensor_dip, sensor_azi = get_sensor_directions_from_channel()
        
        # all combinations of traces in stream:
        # - same number of samples, same start/endtime
        for tr_1 in st:
            for tr_2 in st:
                if tr_1 != tr_2:
                    
                    # ensure same number of samples
                    if not(tr_1.stats.npts == tr_2.stats.npts):
                        tr_1.data = tr_1.data[0:tr_2.stats.npts]
                    
                    # trim start/end
                    tr_1.trim(
                        starttime=tr_2.stats.starttime, 
                        endtime=tr_2.stats.endtime,
                        nearest_sample=\
                            constants.WAVEFORM_READ_ROTATE_NEAREST_SAMPLE)
        
        # init new stream for ZNE directions
        st_ZNE = obspy.Stream()

        if len(st.select(channelnames['U'])) > 0 and \
            len(st.select(channelnames['U'])) > 0 and \
            len(st.select(channelnames['W'])) > 0:
            
            # ObsPy rotate2zne()
            data_ZNE = rotate2zne(
                st.select(channel=channelnames['U'])[0].data, 
                sensor_azi.get('azi_u'), sensor_dip.get('dip_u'),
                
                st.select(channel=channelnames['V'])[0].data,
                sensor_azi.get('azi_v'), sensor_dip.get('dip_v'),
                
                st.select(channel=channelnames['W'])[0].data,
                sensor_azi.get('azi_w'), sensor_dip.get('dip_w'))
            
            for channel_char, data in zip(
                constants.CHANNEL_ZNE_CODES, data_ZNE):
                
                tr = st.select(channel=channelnames['U'])[0].copy()
                
                tr.stats.channel = "{}{}".format(chan_name, channel_char)
                tr.data = data
                st_ZNE += tr
    
    return st_ZNE


def get_sensor_directions_from_channel(chan_u, chan_v, chan_w):
            
    sensor_dip = dict(
        dip_u=chan_u.dip, dip_v=chan_v.dip, dip_w=chan_w.dip)
    sensor_azi = dict(
        azi_u=chan_u.azimuth, azi_v=chan_v.azimuth, 
        azi_w=chan_w.azimuth)
    
    return sensor_dip, sensor_azi
        
        
def read_data(
    fnam_complete, 
    inv, 
    kind, 
    twin, 
    fmin=constants.WAVEFORM_READ_FILTER_HIGHPASS_FMIN, 
    station=constants.DEFAULT_STATION_NAME,
    location_code=constants.DEFAULT_LOCATION_CODE,
    remove_response=True):
    
    # read and process stream if file exists 
    if len(glob.glob(fnam_complete)) > 0:
        
        starttime = twin[0] - \
            constants.WAVEFORM_READ_INITIAL_START_END_TIME_MARGIN
        
        endtime = twin[1] + \
            constants.WAVEFORM_READ_INITIAL_START_END_TIME_MARGIN
        
        st = obspy.read(
            fnam_complete, starttime=starttime, endtime=endtime,
            nearest_sample=constants.WAVEFORM_READ_INITIAL_NEAREST_SAMPLE)
        
        st_seis = st.select(channel='?[LH]?')
        
        # if we have more than one traces, merge them into one
        st_seis.merge(method=1, fill_value='interpolate')
        
        # filter trace 
        # - (1) ObsPy detrend/demean
        # - (2) ObsPy taper(0.1)
        # - (3) ObsPy filter highpass, zerophase, 0.5 * fmin (fmin=0.05)
        # - (4) ObsPy detrend
        
        st_seis.detrend(type=constants.WAVEFORM_READ_FILTER_DETREND_DEMEAN)
        st_seis.taper(
            constants.WAVEFORM_READ_TAPER_MAX_PERCENTAGE, 
            type=constants.WAVEFORM_READ_TAPER_TYPE)
        
        st_seis.filter(
            btype=constants.WAVEFORM_READ_FILTER_BAND_TYPE, 
            ftype=constants.WAVEFORM_READ_FILTER_FREQ_TYPE,
            zerophase=constants.WAVEFORM_READ_FILTER_HIGHPASS_ZP, 
            freq=0.5*fmin)
        
        st_seis.detrend(type=constants.WAVEFORM_READ_FILTER_DETREND_SIMPLE)
        
        if len(st_seis) > 0:
            
            # correct subsample shift for traces earlier than a certain
            # timestamp
            if st_seis[0].stats.starttime < utct(
                constants.WAVEFORM_READ_SUBSAMPLE_SHIFT_CORRECTION_BEFORE):
                
                correct_shift(st_seis.select(channel='??U')[0], nsamples=-1)
            
            # pre-filter, remove response
            for tr in st_seis:
                fmax = tr.stats.sampling_rate * 0.5
                pre_filt = (0.5 * fmin, fmin, fmax * 1.2, fmax * 1.5)
                
                if remove_response:
                    remove_response_stable(
                        tr, inv, output=kind, pre_filt=pre_filt)

            st_rot = create_ZNE_HG(
                st_seis, inv=inv, station=station, location_code=location_code)
            
            if len(st_rot) > 0:
                
                # dig out MHZ channel, special treatment
                if st_rot.select(channel='??Z')[0].stats.channel == 'MHZ':
                    
                    # fnam = fnam_complete[0:-32] + 'BZC' + \
                    #        fnam_complete[-29:-17] + \
                    #        '58.BZC' + fnam_complete[-11:]
                       
                    fnam = "{}{}{}{}{}".format(
                        fnam_complete[0:-32], 'BZC', fnam_complete[-29:-17],
                        '58.BZC', fnam_complete[-11:])
                    
                    starttime = twin[0] - \
                        constants.WAVEFORM_READ_TIME_MARGIN_MHZ, 
                    endtime = twin[1] + \
                        constants.WAVEFORM_READ_TIME_MARGIN_MHZ
                        
                    tr_Z = obspy.read(
                        fnam, starttime=starttime, 
                        endtime=twin[1] + endtime)[0]
                    
                    fmax = tr_Z.stats.sampling_rate * 0.45
                    
                    if remove_response:
                        tr_Z.remove_response(
                            inv, 
                            pre_filt=(
                                constants.WAVEFORM_READ_MHZ_PRE_FILT_FMIN_1, 
                                constants.WAVEFORM_READ_MHZ_PRE_FILT_FMIN_2,
                            fmax, fmax * 1.2), output=kind)
                        
                    st_tmp = st_rot.copy()
                    st_rot = obspy.Stream()
                    
                    tr_Z.stats.channel = 'MHZ'
                    st_rot += tr_Z
                    st_rot += st_tmp.select(channel='?HN')[0]
                    st_rot += st_tmp.select(channel='?HE')[0]

                try:
                    
                    # set 'NaN' values in traces to 0.0
                    for tr in st_rot:
                        tr.data[np.isnan(tr.data)] = 0.0
                    
                    st_rot.filter(
                        btype=constants.WAVEFORM_READ_FILTER_BAND_TYPE, 
                        ftype=constants.WAVEFORM_READ_FILTER_FREQ_TYPE,
                        zerophase=constants.WAVEFORM_READ_FILTER_HIGHPASS_ZP, 
                        freq=fmin)
                
                except(NotImplementedError):
                    # if there are gaps in the stream, return empty stream
                    st_rot = obspy.Stream()
                
                else:
                    # trim traces to original time window
                    st_rot.trim(starttime=twin[0], endtime=twin[1])
        
        else:
            st_rot = obspy.Stream()
    
    else:
        # return empty stream
        st_rot = obspy.Stream()
    
    return st_rot


def remove_response_stable(tr, inv, **kwargs):
    
    try:
        tr.remove_response(inv, **kwargs)
    
    except ValueError:
        
        filtered_inv = inv.select(
            location=tr.stats.location, 
            channel=tr.stats.channel,
            starttime=tr.stats.starttime - \
                constants.WAVEFORM_READ_RESPONSE_FILTERED_TIME_MARGIN,
            endtime=tr.stats.endtime + \
                constants.WAVEFORM_READ_RESPONSE_FILTERED_TIME_MARGIN)

        if filtered_inv:
            last_epoch = filtered_inv[0][0][0]
            last_epoch.start_date = tr.stats.starttime - \
                constants.WAVEFORM_READ_RESPONSE_LASTEPOCH_TIME_MARGIN
            last_epoch.end_date = tr.stats.endtime + \
                constants.WAVEFORM_READ_RESPONSE_LASTEPOCH_TIME_MARGIN

            tr.remove_response(inventory=filtered_inv, **kwargs)
        
        else:
            raise ValueError


def remove_sensitivity_stable(tr, inv, **kwargs):
    try:
        tr.remove_sensitivity(inv, **kwargs)
    
    except ValueError:
        
        filtered_inv = inv.select(
            location=tr.stats.location, channel=tr.stats.channel,
            starttime=tr.stats.starttime - \
                constants.WAVEFORM_READ_RESPONSE_FILTERED_TIME_MARGIN,
            endtime=tr.stats.endtime + \
                constants.WAVEFORM_READ_RESPONSE_FILTERED_TIME_MARGIN)

        if filtered_inv:
            last_epoch = filtered_inv[0][0][0]
            last_epoch.start_date = tr.stats.starttime - \
                constants.WAVEFORM_READ_RESPONSE_LASTEPOCH_TIME_MARGIN
            last_epoch.end_date = tr.stats.endtime + \
                constants.WAVEFORM_READ_RESPONSE_LASTEPOCH_TIME_MARGIN

            tr.remove_sensitivity(inventory=filtered_inv, **kwargs)
        else:
            raise ValueError


def correct_subsample_shift(st):
    """
    Seems to be DEPRECATED, replaced by correct_shift()
    """
    
    if len(st) > 1:
        shift = np.zeros(3)
        for i in range(1, 3):
            shift[i] = (st[i].stats.starttime - st[0].stats.starttime) % \
                       st[0].stats.delta

        if shift.sum() > 0.01:
            starttime = utct(0)
            endtime = utct()
            for tr in st:
                starttime = utct(max(float(starttime),
                                     float(tr.stats.starttime)))
                endtime = utct(min(float(endtime),
                                   float(tr.stats.endtime)))
            print(st)
            st.resample(tr.stats.sampling_rate * 10, no_filter=True)
            print(st)
            st.trim(starttime=starttime, endtime=endtime)
            print(st)
            st.decimate(5, no_filter=True)
            st.decimate(2, no_filter=True)
            print(st)


def correct_shift(tr, nsamples=-1):
    
    ltrace = tr.stats.npts
    
    if nsamples < 0:
        tr.data[0:ltrace + nsamples] = tr.data[-nsamples:ltrace]
    
    elif nsamples > 0:
        tr.data[nsamples:ltrace] = tr.data[0:ltrace - nsamples]
    
    return True


def __dayplot_set_x_ticks(ax, starttime, endtime, sol=False):
    """
    Sets the xticks for the dayplot.
    """

    # day_break = endtime - float(endtime) % 86400
    # day_break -= float(day_break) % 1
    hour_ticks = []
    ticklabels = []
    interval = endtime - starttime
    interval_h = interval / 3600.
    ts = starttime
    tick_start = utct(ts.year, ts.month, ts.day, ts.hour)

    step = 86400
    if 0 < interval <= 60:
        step = 10
    elif 60 < interval <= 300:
        step = 30
    elif 300 < interval <= 900:
        step = 120
    elif 900 < interval <= 1800:
        step = 300
    elif 1800 < interval <= 7200:
        step = 600
    elif 7200 < interval <= 18000:
        step = 1800
    elif 18000 < interval <= 43200:
        step = 3600
    elif 43200 < interval <= 86400:
        step = 4 * 3600
    elif 86400 < interval:
        step = 6 * 3600
    step_h = step / 3600.

    # make sure the start time is a multiple of the step
    if tick_start.hour % step_h > 0:
        tick_start += 3600 * (step_h - tick_start.hour % step_h)

    # for ihour in np.arange(0, interval_h + step_h * 2, step_h):
    for ihour in np.arange(0, interval_h + 2 + step_h, step_h):
        hour_tick = tick_start + ihour * 3600.
        hour_ticks.append(hour_tick)
        if sol:
            ticklabels.append(utct(hour_tick).strftime('%H:%M:%S%nSol %j'))
        else:
            ticklabels.append(utct(hour_tick).strftime('%H:%M:%S%n%Y-%m-%d'))

    hour_ticks_minor = []
    for ihour in np.arange(0, interval_h, 1):
        hour_tick = tick_start + ihour * 3600.
        hour_ticks_minor.append(hour_tick)

    ax.set_xlim(float(starttime),
                float(endtime))
    ax.set_xticks(hour_ticks)
    ax.set_xticks(hour_ticks_minor, minor=True)
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(float(starttime),
                float(endtime))


def calc_PSD(tr, winlen_sec, detick_nfsamp=0, padding=True):
    Fs = tr.stats.sampling_rate

    if detick_nfsamp > 0:
        tr = detick(tr, detick_nfsamp)

    winlen = min(winlen_sec * Fs,
                 (tr.stats.endtime - tr.stats.starttime) * Fs / 2.)
    NFFT = next_pow_2(winlen)
    if padding:
        pad_to = np.max((NFFT * 2, 1024))
    else:
        pad_to = NFFT
    p, f = mlab.psd(tr.data,
                    Fs=Fs, NFFT=NFFT, detrend='linear',
                    pad_to=pad_to, noverlap=NFFT // 2)
    return f, p


def detick(tr, detick_nfsamp, fill_val=None, freq_tick=1.0):
    # simplistic deticking by muting detick_nfsamp freqeuency samples around
    # 1Hz
    tr_out = tr.copy()
    Fs = tr.stats.sampling_rate
    NFFT = next_pow_2(tr.stats.npts)
    tr.detrend()
    df = np.fft.rfft(tr.data, n=NFFT)
    idx_1Hz = np.argmin(np.abs(np.fft.rfftfreq(NFFT) * Fs - freq_tick))
    if fill_val is None:
        fill_val = (df[idx_1Hz - detick_nfsamp - 1] + \
                    df[idx_1Hz + detick_nfsamp + 1]) / 2.
    df[idx_1Hz - detick_nfsamp:idx_1Hz + detick_nfsamp] /= \
        df[idx_1Hz - detick_nfsamp:idx_1Hz + detick_nfsamp] / fill_val
    tr_out.data = np.fft.irfft(df)[:tr.stats.npts]
    return tr_out


def plot_spectrum(ax, ax_all, df_mute, iax, ichan_in, spectrum,
                  fmin=0.1, fmax=100.,
                  **kwargs):
    f = spectrum['f']
    for i, chan in enumerate(['Z', 'N', 'E']):
        ichan = ichan_in + i
        try:
            p = spectrum['p_' + chan]
        except(KeyError):
            continue
        else:
            bol_1Hz_mask = np.array(
                (np.array((f > fmin, f < fmax)).all(axis=0),
                 np.array((f < 1. / df_mute,
                           f > df_mute)).any(axis=0))
                ).all(axis=0)

            bol_1Hz_mask = np.invert(bol_1Hz_mask)
            p = np.ma.masked_where(condition=bol_1Hz_mask, a=p,
                                   copy=False)
            f = np.ma.masked_where(condition=bol_1Hz_mask, a=f,
                                   copy=False)

            if ichan % 3 == 0:
                ax_all[ichan % 3].plot(f,
                                       10 * np.log10(p),
                                       lw=0.5, c='lightgrey', zorder=1)
                ax[iax, ichan].plot(f,
                                    10 * np.log10(p),
                                    **kwargs)
            elif ichan % 3 == 1:
                tmp2 = p
            elif ichan % 3 == 2:
                ax_all[ichan % 3 - 1].plot(f,
                                           10 * np.log10(tmp2 + p),
                                           lw=0.5, c='lightgrey', zorder=1)
                ax[iax, ichan - 1].plot(f,
                                        10 * np.log10(p + tmp2),
                                        **kwargs)

            # ax[iax, ichan].axes.get_xaxis().set_visible(False)
            # ax[iax, ichan].axes.get_yaxis().set_visible(False)
            ichan += 1


def envelope_smooth(envelope_window_in_sec, tr, mode='valid'):
    
    tr_env = tr.copy()
    tr_env.data = envelope(tr_env.data)

    w = np.ones(int(envelope_window_in_sec / tr.stats.delta))
    w /= w.sum()
    tr_env.data = np.convolve(tr_env.data, w, mode=mode)

    return tr_env


# Autocorrelation stuff

# def norm_hilbert(x, Fs):
#     x_white = whiten(x)
#     x_filt = filt(x_white, Fs=Fs, freqs=(1.5, 4.))
#     Z = hilbert(x_filt)
#     return Z / np.abs(Z)


def whiten(x):
    fx = fft(x, n=next_pow_2(len(x)))
    fx /= np.abs(fx)
    return ifft(fx, n=next_pow_2(len(x))).real[0:len(x)]


def inst_phase(x):
    Z = hilbert(x)
    return np.angle(Z)


def filt(x, Fs, freqs):
    from scipy.signal import filtfilt, butter

    b, a = butter(N=8, Wn=freqs[0] / (Fs / 2), btype='high')
    y = filtfilt(b, a, x)
    b, a = butter(N=8, Wn=freqs[1] / (Fs / 2), btype='low')
    y = filtfilt(b, a, y)
    return y


def phase_ac(x, Fs, maxlag_sec=8., nu=2.5):
    # Phase cross-correlation as defined in
    # Schimmel, M.(1999), Phase cross-correlations: design, comparisons and
    # applications, Bull.Seismol.Soc.Am., 89, 1366 - -1378.
    # This function implements eq. 4
    # Parameter nu was introduced later, e.g. eq. 2 in:
    # Schimmel, M., E. Stutzmann, and J. Gallart (2011), Using instantaneous
    # phase coherence for signal extraction from ambient noise data at a
    # local to a global scale,
    # Geophys. J. Int., 184, 494–506, doi:10.1111/j.1365-246X.2010.04861.x.
    maxlag = int(maxlag_sec * Fs)
    ac = np.zeros(maxlag)
    i = 0
    for ilag in range(1, maxlag):  # -maxlag//2, maxlag//2):
        # plusterm = np.abs(norm_hilbert(x[0:-ilag], Fs)
        #                   + norm_hilbert(x[ilag:], Fs))
        # minusterm = np.abs(norm_hilbert(x[0:-ilag], Fs)
        #                    - norm_hilbert(x[ilag:], Fs))

        A = np.exp(1.j * inst_phase(x[0:-ilag]))
        B = np.exp(1.j * inst_phase(x[ilag:]))
        plusterm = np.abs(A + B)
        minusterm = np.abs(A - B)

        ac[i] = 1. / (2 * len(x)) * np.sum(plusterm ** nu - minusterm ** nu) * \
                np.sqrt(ilag / Fs)
        i += 1
    return ac


def autocorrelation(st, starttime, endtime, fmin=1.2, fmax=3.5, max_lag_sec=40):
    # st.decimate(2)

    Fs = int(st[0].stats.sampling_rate)
    max_lag = max_lag_sec * Fs

    fig, ax = plt.subplots(nrows=4, ncols=1, sharey='all', sharex='all',
                           figsize=(15, 8))

    freqs = [[1.1, 3.5],
             [1.1, 5.0],
             [1.1, 8.0],
             [3.0, 6.0]]
    for i, freq in enumerate(freqs):
        print(freq)
        st_work = st.copy()
        st_work.filter('highpass', freq=1. / 10., zerophase=True)
        st_work.filter('lowpass', freq=8., zerophase=True)
        st_work.trim(starttime=starttime,
                     endtime=endtime)
        st_work.taper(max_percentage=0.05)
        acsum = np.zeros((max_lag, 4))
        for tr in st_work:
            data = whiten(tr.data)
            data = filt(data, Fs=tr.stats.sampling_rate,
                        freqs=(freq[0], freq[1]))
            ac = phase_ac(data,
                          Fs=tr.stats.sampling_rate,
                          maxlag_sec=max_lag_sec)
            t_ac = np.arange(0, len(ac)) / Fs
            ax[i].plot(t_ac,
                       filt(ac, Fs=tr.stats.sampling_rate,
                            freqs=(fmin, fmax)),
                       lw=2, label=tr.stats.channel)
            acsum[:, i] += filt(ac, Fs=tr.stats.sampling_rate,
                                freqs=(fmin, fmax))

            # ac_CC = np.correlate(tr.data, tr.data, mode='same') \
            #         / (np.sum(tr.data * tr.data))
            # ax[1].plot(np.arange(-len(ac_CC) / 2, len(ac_CC) / 2) / Fs,
            #            ac_CC, lw=2, label=tr.stats.channel)
            # acsum_CC += ac_CC
        ax[i].plot(t_ac, acsum[:, i], lw=2, c='k',
                   label='Sum')
    # ax[0].plot(np.arange(0, len(acsum)) / Fs, abs(hilbert(acsum)),
    #            label='Env. of Sum',
    #            lw=2, c='r')
    ax[0].legend()
    # ax[1].plot(np.arange(-len(acsum_CC) / 2, len(acsum_CC) / 2) / Fs,
    #            acsum_CC, lw=2, c='k')
    # ax[1].plot(np.arange(-len(acsum_CC) / 2, len(acsum_CC) / 2) / Fs,
    #            abs(hilbert(acsum_CC)), lw=2, c='r')
    ax[1].set_xlabel('seconds')
    ax[0].set_ylim(-1.2, 1.2)
    ax[1].set_ylim(-1.2, 1.2)
    ax[0].set_xlim((0, 20))
    for a in ax:
        a.set_xticks(np.arange(0, 30), minor=True)
        a.grid('on', which='major')
        a.grid('on', which='minor', ls='dashed', color='grey')

    ax[0].set_title('Phase autocorrelation')
    # ax[1].set_title('CC autocorrelation')
    return fig, ax


def source_spec(f: np.array,
                f_c: np.array,
                n=2.,
                gamma=2.
                ):
    """
    Compute source spectrum of general Boatwright form
    :param f: frequency in Hz
    :param f_c: corner frequency in Hz
    :param n: Parameter 1, ==2 for Brune and Boatwright source
    :param gamma: 1 for Brune, 2 for Boatwright
    :return:
    """
    omega = f * 2. * np.pi
    omega_c = f_c * 2. * np.pi
    # equation 7 in Bostock et al. 2017
    denom = (1. + 1. / (n - 1.) * (omega / omega_c) ** (n * gamma)) ** (
                1. / gamma)

    return 1. / denom


def att_spec(f: np.array,
             tstar: float):
    omega = f * 2. * np.pi
    # equation 9 in Bostock et al. 2017
    return np.exp(-omega * tstar / 2.)


def att_spec_fdef(f: np.array,
                  tstar0: float,
                  alpha=0.25):
    omega = f * 2. * np.pi
    # equation 9 in Bostock et al. 2017
    return np.exp(-omega * tstar0 * f ** (-alpha) / 2.)


def complete_spec(f, A0, tstar, f_c):
    spec = att_spec(f, tstar) * source_spec(f, f_c) * A0 * 2. * np.pi * f * 1e-9
    return spec


def complete_spec_fdef(f, A0, tstar0, f_c):
    spec = att_spec_fdef(f, tstar0) * \
           source_spec(f, f_c) * A0 * 2. * np.pi * f * 1e-9
    return spec


def spectral_fit(f: np.array,
                 p_signal: np.array,
                 sigma_signal: np.array,
                 p_noise: np.array,
                 fnam: str,
                 fmin: float,
                 fmax: float):
    
    from scipy.optimize import curve_fit
    
    fit_bol = np.array((f > float(fmin),
                        f < float(fmax))).all(axis=0)

    #sigma_signal = np.ones_like(p_signal) * 1e-10
    signal_red = p_signal ** 2 - p_noise ** 2
    signal_red[signal_red < 0.] = 0.
    signal_red = np.sqrt(signal_red)
    
    popt, pcov = curve_fit(complete_spec,
                           f[fit_bol],
                           signal_red[fit_bol],
                           sigma=sigma_signal[fit_bol],
                           # p_signal - p_noise,
                           bounds=((0., 0.2, 0.1), (20, 3., 3.)),
                           p0=(1., 1., 1.2))
    # , A0, tstar, f_c)

    # plt.figure()
    # plt.plot(f, complete_spec(f, popt[0], popt[1], popt[2]),
    #          label='fit, $t^*$=%3.1fs, $f_c$=%3.1fHz' % (popt[1], popt[2]))
    # plt.errorbar(f, p_signal, yerr=sigma_signal, label='data', c='C5')
    # # plt.plot(f, p_signal, label='data', c='C5')
    # plt.plot(f, p_signal - p_noise, label='data - noise', c='C5', ls='dashed')
    # plt.plot(f, signal_red,
    #          label='data - noise, squared', c='C5', ls='dotted')
    # plt.plot(f, p_noise, label='noise')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(1e-11, 2e-8)
    # plt.legend()
    # print(popt)
    # print(pcov)
    # plt.savefig(fnam)
    return popt[0], popt[1], popt[2], pcov[1, 1]


def calc_mt_spec(tr, t_ref, tmin_amp, tmax_amp):
    
    from mqs_reports.utils import detick
    import mtspec

    tr_detick = detick(tr, detick_nfsamp=5)

    tr_amp = tr_detick.slice(
        starttime=t_ref + tmin_amp, endtime=t_ref + tmax_amp)
    
    res = mtspec.mtspec(
        data=tr_amp.data, delta=tr_amp.stats.delta, time_bandwidth=2.5,
        statistics=True)
    
    f = res[1]
    p = np.sqrt(res[0])
    p_low = np.sqrt(res[2][:, 0])
    p_up = np.sqrt(res[2][:, 1])
    return f, p, p_low, p_up


def linregression(x: np.array, y: np.array, q: float = 0.95) -> tuple:
    # Do a linear regression for value pairs X, Y and return error estimate
    # for slope and intercept
    
    from scipy import stats
    
    n = len(x)
    slope, intercept, r_value, p_value, slope_err = stats.linregress(x, y)

    intercept_err = slope_err * np.sqrt(1. / n * np.sum(x * x))

    tstar = stats.t.ppf(q=q, df=n - 2)

    return (intercept, intercept_err * tstar, slope, slope_err * tstar)


def calc_specgram(tr, fmin=1. / 50, fmax=1. / 2, w0=16):
    
    from matplotlib.mlab import specgram
    
    dt = tr.stats.delta

    s, f, t = specgram(
        x=tr.data, NFFT=512, Fs=tr.stats.sampling_rate, noverlap=256, 
        pad_to=1024)

    t = create_timevector(tr)
    f_bol = np.asarray(((fmin < f),
                        (f < fmax))).all(axis=0)

    return s[f_bol, :], f[f_bol], t


def calc_cwf(tr, fmin=1. / 50, fmax=1. / 2, w0=16):
    
    from obspy.signal.tf_misfit import cwt
    
    dt = tr.stats.delta

    scalogram = abs(
        cwt(tr.data, dt, w0=w0, nf=200, fmin=fmin, fmax=fmax))

    t = create_timevector(tr)
    # t = np.linspace(0, dt * tr.stats.npts, tr.stats.npts)
    f = np.logspace(np.log10(fmin),
                    np.log10(fmax),
                    scalogram.shape[0])
    
    return scalogram ** 2, f, t


def create_timevector(tr):
    
    timevec = [
        utct(t + float(tr.stats.starttime)).datetime for t in tr.times()]
    
    return timevec


def uncertainty_from_pdf(variable: np.array, p: np.array):
    """
    Fit a Gaussian through the pdf expressed by variable, p
    From Savas Ceylan
    """
    
    from scipy.interpolate import UnivariateSpline
    
    spline = UnivariateSpline(variable, p - np.nanmax(p) / 4, s=0)
    _roots = spline.roots()
    
    # print(_roots, variable[p == np.nanmax(p)], np.diff(_roots) / 2.)

    if len(_roots) > 1:
        r1 = _roots[0]
        r2 = _roots[1]
    else:
        r1 = 0
        r2 = 30

    return r1, r2


def add_orientation_to_stream_info(stream_info, orientation):
    
    stream_info_split = stream_info.split('@')
    
    stream_info_split_1 = stream_info_split[0]
    stream_info_split_2 = ""
    
    if len(stream_info_split) > 1:
        stream_info_split_2 = stream_info_split[1]
        
    stream_info_with_orientation = "{}{}@{}".format(
        stream_info_split_1, orientation, stream_info_split_2)
    
    return stream_info_with_orientation


def get_streaminfo(event, component=None):
    
    streaminfo = dict(
        LF_streaminfo="", LF_streaminfo_with_orientation="", 
        HF_streaminfo="", HF_streaminfo_with_orientation="")
    
    if event.spectra is None:
        return streaminfo
    
    if 'stream_info' in event.spectra and \
            event.spectra['stream_info'].startswith("LF"):
        streaminfo['LF_streaminfo'] = event.spectra['stream_info']
        
    if 'stream_info' in event.spectra_SP and \
            event.spectra_SP['stream_info'].startswith("HF"):
        streaminfo['HF_streaminfo'] = event.spectra_SP['stream_info']
    
    if streaminfo['LF_streaminfo'] and component is not None:
        streaminfo['LF_streaminfo_with_orientation'] = \
            add_orientation_to_stream_info(
                streaminfo['LF_streaminfo'], component)
        
    if streaminfo['HF_streaminfo'] and component is not None:
        streaminfo['HF_streaminfo_with_orientation'] = \
            add_orientation_to_stream_info(
                streaminfo['HF_streaminfo'], component)
    
    return streaminfo
            

def get_stream_from_instrument(event, instrument):
    
    if instrument == 'VBB':
        stream = event.waveforms_VBB.copy()
        
    elif instrument == 'SP':
        stream = event.waveforms_SP.copy()
        
    elif instrument == 'VBB100':
        stream = event.waveforms_VBB100.copy()
        
    elif instrument == 'VBB+VBB100':
        stream = event.waveforms_VBB100.copy()
        
    elif instrument == 'VBB+SP':
        stream = event.waveforms_SP.copy()
        
    else:
        raise ValueError(f'Invalid value for instrument: {instrument}')
    
    return stream 
   
   
def set_streaminfo_plot(
    event, streaminfo, fmin, fmax, lf=None, hf=None, ref_time=None, 
    ref_time_type=None, wf_type="RAW"):

    streaminfo_plot = dict(
        LF=lf, 
        HF=hf, 
        event_name=event.name, 
        origin_time_screen=event.origin_time_screen,
        origin_time=event.origin_time,
        mars_event_type_short=event.mars_event_type_short,
        location_quality_1char=event.quality,
        fmin=fmin, 
        fmax=fmax, 
        ref_time=ref_time,
        ref_time_type=ref_time_type,
        p_pick_time_screen=None,
        wf_type=wf_type)
    
    if ref_time is not None:
        streaminfo_plot['p_pick_time_screen'] = \
            marsutils.get_rounded_timestamps(ref_time).get(
                'TIMESTAMP_READABLE_FORMAT')

    LF_streaminfo = None 
    LF_streaminfo_with_orientation = None
    HF_streaminfo = None 
    HF_streaminfo_with_orientation = None 
    
    if lf is None:
        LF_streaminfo = streaminfo.get("LF_streaminfo")
        LF_streaminfo_with_orientation = streaminfo.get(
            "LF_streaminfo_with_orientation")
        
    if hf is None:
        HF_streaminfo = streaminfo.get("HF_streaminfo")
        HF_streaminfo_with_orientation = streaminfo.get(
            "HF_streaminfo_with_orientation")
        
    if LF_streaminfo is not None:
        streaminfo_plot["LF"] = LF_streaminfo
        streaminfo_plot["LF_orientation"] = LF_streaminfo_with_orientation

    if HF_streaminfo is not None:
        streaminfo_plot["HF"] = HF_streaminfo
        streaminfo_plot["HF_orientation"] = HF_streaminfo_with_orientation
    
    return streaminfo_plot


def get_stream_description(event, instrument):
        
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
        raise ValueError(
            "get_stream_description: invalid value for instrument: {}".format(
                instrument))
    
    return st_LF, st_HF, st_LF_desc, st_HF_desc
    
    
def get_color_for_marker(phase):
    
    if phase in ('P', 'PP'):
        markercolor = constants.COLOR_FILTERBANK_P_PHASE
    elif phase in ('S', 'SS'):
        markercolor = constants.COLOR_FILTERBANK_S_PHASE
    else:
        markercolor = constants.COLOR_FILTERBANK_START_END_PHASE
        
    return markercolor


def mark_glitch(
    ax: list, 
    x0: float, 
    x1: float, 
    ymin: float=constants.FILTERBANK_PLOT_GLITCH_YMIN_DEFAULT, 
    height: float=constants.FILTERBANK_PLOT_GLITCH_HEIGHT_DEFAULT, 
    **kwargs):
    """
    Mark glitch in plot with rectangle box.
    
    """
    
    xy = [x0, ymin]
    width = x1 - x0
    
    for a in ax:
        rect = Rectangle(xy=xy, width=width, height=height, **kwargs)
        a.add_patch(rect)

                
def add_suptitle_textboxes(ax, streaminfo_plot, kind='spectra'):
    """
    Add info text boxes below plot main title.
    
    """
    
    if kind == 'spectra':
        tbx_1 = constants.SPECTRA_TEXT_BOXES_XCOORD['type_quality']
        tbx_2 = constants.SPECTRA_TEXT_BOXES_XCOORD['origin_time']
        tbx_4 = constants.SPECTRA_TEXT_BOXES_XCOORD['raw_denoised_deglitched']
        tbx_5 = constants.SPECTRA_TEXT_BOXES_XCOORD['streamid_lf']
        tbx_6 = constants.SPECTRA_TEXT_BOXES_XCOORD['streamid_hf']
        tbx_7 = constants.SPECTRA_TEXT_BOXES_XCOORD['filtercode']
        
        lf_box_key = 'LF_orientation'
        hf_box_key = 'HF_orientation'
        
    elif kind == 'filterbanks':
        tbx_1 = constants.FILTERBANK_TEXT_BOXES_XCOORD['type_quality']
        tbx_2 = constants.FILTERBANK_TEXT_BOXES_XCOORD['origin_time']
        tbx_3 = constants.FILTERBANK_TEXT_BOXES_XCOORD['p_phase_time']
        tbx_4 = \
            constants.FILTERBANK_TEXT_BOXES_XCOORD['raw_denoised_deglitched']
        tbx_5 = constants.FILTERBANK_TEXT_BOXES_XCOORD['streamid_lf']
        tbx_6 = constants.FILTERBANK_TEXT_BOXES_XCOORD['streamid_hf']
        tbx_7 = constants.FILTERBANK_TEXT_BOXES_XCOORD['freq_range']
        
        lf_box_key = 'LF'
        hf_box_key = 'HF'
        
    
    # top text boxes below suptitle
    ax.text(
        tbx_1, 
        constants.SPECTRA_TEXT_BOXES_YCOORD, 
        "{}/Q{}".format(
            streaminfo_plot["mars_event_type_short"], 
            streaminfo_plot["location_quality_1char"]),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, bbox=constants.SPECTRA_TEXT_BOX_PARAMS,
        color=constants.COLOR_SPECTRA_TOP_TEXT_BOXES, 
        fontsize=constants.SPECTRA_TEXT_BOXES_FONTSIZE)
    
    ax.text(
        tbx_2, 
        constants.SPECTRA_TEXT_BOXES_YCOORD, 
        "OT: {}".format(streaminfo_plot["origin_time_screen"]),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, bbox=constants.SPECTRA_TEXT_BOX_PARAMS,
        color=constants.COLOR_SPECTRA_TOP_TEXT_BOXES, 
        fontsize=constants.SPECTRA_TEXT_BOXES_FONTSIZE)
    
    ax.text(
        tbx_4, 
        constants.SPECTRA_TEXT_BOXES_YCOORD, 
        "{} waveforms".format(streaminfo_plot["wf_type"].lower()),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, bbox=constants.SPECTRA_TEXT_BOX_PARAMS,
        color=constants.COLOR_SPECTRA_TOP_TEXT_BOXES, 
        fontsize=constants.SPECTRA_TEXT_BOXES_FONTSIZE)
        
    if streaminfo_plot["LF"] is not None:
        ax.text(
            tbx_5, 
            constants.SPECTRA_TEXT_BOXES_YCOORD, 
            streaminfo_plot[lf_box_key],
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes, bbox=constants.SPECTRA_TEXT_BOX_PARAMS,
            color=constants.COLOR_SPECTRA_TOP_TEXT_BOXES, 
            fontsize=constants.SPECTRA_TEXT_BOXES_FONTSIZE)
    
    if streaminfo_plot["HF"] is not None:
        ax.text(
            tbx_6, 
            constants.SPECTRA_TEXT_BOXES_YCOORD, 
            streaminfo_plot[hf_box_key],
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes, bbox=constants.SPECTRA_TEXT_BOX_PARAMS,
            color=constants.COLOR_SPECTRA_TOP_TEXT_BOXES, 
            fontsize=constants.SPECTRA_TEXT_BOXES_FONTSIZE)
    
    if kind == 'spectra':
        filterbox_text_sub = constants.SPECTRA_PLOT_FILTER_BOX_LABEL_1 + \
            str(constants.WAVEFORM_READ_FILTER_HIGHPASS_FMIN) + \
            constants.SPECTRA_PLOT_FILTER_BOX_LABEL_2
        
        ax.text(
            tbx_7, 
            constants.SPECTRA_TEXT_BOXES_YCOORD, 
            filterbox_text_sub, verticalalignment='center', 
            horizontalalignment='center',
            transform=ax.transAxes, bbox=constants.SPECTRA_TEXT_BOX_PARAMS,
            color=constants.COLOR_SPECTRA_TOP_TEXT_BOXES, 
            fontsize=constants.SPECTRA_TEXT_BOXES_FONTSIZE)
        
    if kind == 'filterbanks':
        
        if streaminfo_plot["p_pick_time_screen"] is not None:
            ax.text(
                tbx_3, 
                constants.FILTERBANK_TEXT_BOXES_YCOORD, 
                "{}: {}".format(
                    streaminfo_plot["ref_time_type"],
                    streaminfo_plot["p_pick_time_screen"]),
                verticalalignment='center', horizontalalignment='center',
                transform=ax.transAxes, 
                bbox=constants.FILTERBANK_TEXT_BOX_PARAMS,
                color=constants.COLOR_FILTERBANK_TOP_TEXT_BOXES, 
                fontsize=constants.FILTERBANK_TEXT_BOXES_FONTSIZE)
        
        freq_range_text_sub = "{:5.3f}-{:5.3f} Hz".format(
            streaminfo_plot["fmin"] , streaminfo_plot["fmax"] )
        
        ax.text(
            tbx_7, 
            constants.FILTERBANK_TEXT_BOXES_YCOORD, 
            freq_range_text_sub, verticalalignment='center', 
            horizontalalignment='center',
            transform=ax.transAxes, bbox=constants.FILTERBANK_TEXT_BOX_PARAMS,
            color=constants.COLOR_FILTERBANK_TOP_TEXT_BOXES, 
            fontsize=constants.FILTERBANK_TEXT_BOXES_FONTSIZE)
        
    
