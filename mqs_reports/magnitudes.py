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

import numpy as np

from mqs_reports import constants
from mqs_reports.utils import linregression


def get_Mw(M0):
    return (np.log10(M0) - 9.1) / 1.5


def get_M0(Mw):
    return 10 ** (1.5 * Mw + 9.1)


def calc_magnitude(amplitude_dB: float,
                   distance_degree: float,
                   mag_type: str, version: str,
                   distance_sigma_degree: float,
                   amplitude_sigma_dB: float,
                   verbose=False):
    """

    :param amplitude_dB: Relevant amplitude in dB. Please note that this should
                         be an amplitude, not a power.
                         If it is a power, divide it by 2.
    :param distance_degree: Distance of event
    :param mag_type: allowed values: 'MwspecHF', 'MwspecLF', 'mb_P',
                                     'mb_S', 'm24pick', 'm24spec'
    :param version: 'Giardini2020' or 'Boese2021'
    :param distance_sigma_degree: uncertainty of distance
    :param amplitude_sigma_dB: uncertainty of amplitude in dB
    :return: mag, sigma
    """
    mag_variables = constants.magnitude[version][mag_type]
    amplitude_log = amplitude_dB / 10.
    if amplitude_sigma_dB is None:
        amplitude_sigma_log = 1.
    else:
        amplitude_sigma_log = amplitude_sigma_dB / 10.

    mag = mag_variables['fac'] * (
            amplitude_log +
            mag_variables['ai'] * np.log10(distance_degree) +
            mag_variables['ci']
          )
    if mag_variables['sigma'] is not None:
        sigma = mag_variables['sigma']
    else:
        if distance_sigma_degree is None:
            distance_sigma_log = np.log10(1.2)
        else:
            distance_sigma_log = (np.log10(distance_degree +
                                           0.5 * distance_sigma_degree) -
                                  np.log10(distance_degree -
                                           0.5 * distance_sigma_degree))
        sigma = mag_variables['fac'] * 2. / 3. * \
            np.sqrt(
                    amplitude_sigma_log ** 2. +
                    (np.log10(distance_degree) *
                        mag_variables['ai_sigma']) ** 2. +
                    mag_variables['ai'] ** 2. * distance_sigma_log ** 2. +
                    mag_variables['ci_sigma'] ** 2.
                )
        if verbose:
            print('Sigma: term1: %6.4f, term2: %6.4f, term3: %6.4f, term4: %6.4f, sum: %6.4f' %
                  (4. / 9. * amplitude_sigma_log ** 2.,
                   4. / 9. * np.log10(distance_degree) ** 2. * mag_variables['ai_sigma'] ** 2.,
                   4. / 9. * mag_variables['ai'] ** 2. * distance_sigma_log ** 2.,
                   4. / 9. * mag_variables['ci_sigma'] ** 2.,
                   sigma
                  )
            )

    return mag, sigma


def lorentz(x, A, x0, xw):
    """
    Return a Lorentz peak function centered around x0, with width xw and
    amplitude A in dB values

    Parameters
    ----------
    :param x: x-values to evaluate function at
    :param A: Peak amplitude
    :param x0: center value of peak
    :param xw: width of peak
    :return: Lorentz/Cauchy function in dB
    """
    w = (x - x0) / (xw / 2.)
    return 10. * np.log10(1. / (1. + w ** 2)) + A


def lorentz_modes(x, A, x0, xw, ampfac):
    """
    Return a Lorentz peak function centered around x0, with width xw and
    amplitude A in dB values
    Parameters
    ----------
    :param x: x-values to evaluate function at
    :param A: Peak amplitude - baseline actually I think: gze, 11.12.2020
    :param x0: center value of peak
    :param xw: width of peak
    :return: Lorentz/Cauchy function in dB
    """
    w = (x - x0) / (xw / 2.)
    return A + 10 * np.log10(1 + ampfac / (1 + w ** 2))
    # return A * (1 + ampfac / (1 + w ** 2))
    
    
def lorentz_att(f: np.array,
                A0: float,
                f0: float,
                f_c: float,
                tstar: float,
                fw: float,
                ampfac: float,
                gamma: float = 0.0,
                omega_exp: float = 2.0) -> np.array:
    """
    Attenuation spectrum, combined with Lorentz peak and source spectrum
    :param f: Frequency array (in Hz)
    :param A0: Long-period amplitude in flat part of spectrum (in dB)
    :param f0: Center frequency of the Lorenz peak (aka 2.4 Hz)
    :param f_c: Corner frequency of the Source (Hz)
    :param tstar: t* value from attenuation
    :param fw: Width of Lorentz peak
    :param ampfac: Amplification factor of Lorentz peak (AS FACTOR, NOT IN DB!!!)
    :param gamma: Exponent to rotate the spectral fit
    :param omega_exp: Exponent for the omega term in the spectral fit; (f / fc)^omega_exp
    :return predicted spectrum
    """
    # stf_amp = 1 / (1 + (f / f_c) ** 2)
    # stf_amp = f**gamma / (1. + (f / f_c)** (2 + gamma))
    stf_amp = f**gamma / (1. + (f / f_c)** (omega_exp + gamma))

    w = (f - f0) / (fw / 2.)
    term1 = 1 + ampfac / (1 + w ** 2)
    term2 = stf_amp
    term3 = np.exp(-tstar * f * np.pi)

    result = A0 + 20 * np.log10(term1 * term2 * term3)
    return result

def _remove_singles(array):
    for ix in range(0, len(array) - 1):
        if array[ix] and not (array[ix - 1] or array[ix + 1]):
            array[ix] = False


def fit_peak_att(f, p, A0_max=-200., tstar_min=0.05, A0_min=-240.):
    from scipy.optimize import curve_fit
    tstar_max = 10.0

    # Central frequency of the 2.4 Hz mode (in Hz)
    f0_min = 2.25
    f0_max = 2.5

    # Amplification factor of the 2.4 Hz mode (not in dB!)
    ampfac_min = 10.
    ampfac_max = 400.

    # Width of the 2.4 Hz mode
    fw_min = 0.05
    fw_max = 0.4
    # noinspection PyTypeChecker
    popt, pcov = curve_fit(lorentz_att, f, 10. * np.log10(p),
                           bounds=((A0_min,
                                    f0_min,
                                    0.8,
                                    tstar_min,
                                    fw_min,
                                    ampfac_min),
                                   (A0_max,
                                    f0_max,
                                    10.0,
                                    tstar_max,
                                    fw_max,
                                    ampfac_max)),
                           sigma=f * 10.,
                           p0=((A0_max + A0_min) / 2.,
                               2.4,
                               3.,
                               2.,
                               0.25,
                               ampfac_min))
    return popt


def fit_peak(f, p, A0_min=-240, A0_max=-160,
             f0_min=2.25, f0_max=2.5, fw_min=0.05, fw_max=0.4):
    """
    Fit a spectral peak to function PSD p at frequencies f

    Parameters
    ----------
    :param f: frequency vector [in Hz]
    :param p: power spectral density [in dB]
    :param A0_max: Minimum allowed amplitude for peak
    :param A0_min: Maximum allowed amplitude for peak
    :param f0_min: Minimum allowed frequency for peak
    :param f0_max: Maximum allowed frequency for peak
    :param fw_min: Minimum allowed spectral width [in Hz]
    :param fw_max: Maximum allowed spectral width [in Hz]
    :return: list with Amplitude, central frequency, width of peak
    """
    from scipy.optimize import curve_fit
    try:
        # noinspection PyTypeChecker
        popt, pcov = curve_fit(lorentz, f, 10 * np.log10(p),
                               bounds=((A0_min, f0_min, fw_min),
                                       (A0_max, f0_max, fw_max)),
                               p0=((A0_max + A0_min) * 0.5,
                                   (f0_max + f0_min) * 0.5,
                                   (fw_max + fw_min) * 0.5))
    except ValueError:
        popt = [-250, 2.4, 1.0]

    return popt


def fit_peak_modes(f, p, A0_min=-250, A0_max=-135,
                   f0_min=2.25, f0_max=2.5, fw_min=0.05, fw_max=0.4, 
                   ampfac_min = 100., ampfac_max = 400.): #10, 400
    """
    Fit a spectral peak to function PSD p at frequencies f
    Parameters
    ----------
    :param f: frequency vector [in Hz]
    :param p: power spectral density [in dB]
    :param A0_max: Minimum allowed amplitude for peak
    :param A0_min: Maximum allowed amplitude for peak
    :param f0_min: Minimum allowed frequency for peak
    :param f0_max: Maximum allowed frequency for peak
    :param fw_min: Minimum allowed spectral width [in Hz]
    :param fw_max: Maximum allowed spectral width [in Hz]
    ampfac_min/max: Amplification factor of the mode (not in dB!)
    
    :return: list with Amplitude, central frequency, width of peak
    """
    from scipy.optimize import curve_fit
        # noinspection PyTypeChecker
    try:
        popt, pcov = curve_fit(lorentz_modes, f, p,
                               bounds=((A0_min, f0_min, fw_min, ampfac_min),
                                       (A0_max, f0_max, fw_max, ampfac_max)),
                               p0=((A0_max + A0_min) * 0.5,
                                   (f0_max + f0_min) * 0.5,
                                   (fw_max + fw_min) * 0.5,
                                   (ampfac_max + ampfac_min) * 0.5))

    except RuntimeError:
        popt = [np.nan, np.nan, np.nan, np.nan]
    return popt


def fit_spectra(f_sig, p_sig, f_noise, p_noise, event_type, df_mute=1.05,
                A0_fix=None):
    len_spec = len(f_noise)
    if len(p_sig) != len_spec:
        f_dec = np.linspace(f_noise[0], f_noise[-1], len_spec)
        p_sig = np.interp(x=f_noise, xp=f_sig, fp=p_sig)
    # if len(p_noise) != len_spec:
    #     f_dec = np.linspace(f[0], f[-1], len_spec)
    #     p_noise = np.interp(x=f, xp=f_dec, fp=p_noise)
    f = f_noise
    fmin = 0.1
    fmax = 6.0
    noise_threshold = 2.0
    if event_type == 'LF':
        fmax = 0.9
        # noise_threshold = 1.2
    elif event_type in ['BB', 'WB']:
        fmax = 2.0
        # noise_threshold = 1.2
    elif event_type == 'HF':
        fmin = 1.0
    elif event_type == 'SF':
        fmin = 4.0
        fmax = 9.0

    bol_fitting_mask = np.array(
        (np.array((f > fmin, f < fmax)).all(axis=0),
         np.array((f < 1. / df_mute,
                   f > df_mute)).any(axis=0),
         np.array(p_sig > p_noise * noise_threshold)
         )
    ).all(axis=0)
    _remove_singles(bol_fitting_mask)

    mute_24 = [1.9, 3.4]
    A0 = None
    tstar = None
    A0_err = None
    tstar_err = None
    ampfac = None
    width_24 = None
    f_c = None
    A_24 = None
    f_24 = None
    A_24_red = None
    f_24_red = None
    if event_type not in ['24', 'SF']:
        if sum(bol_fitting_mask) > 5:

            # Fitting HF family events
            if event_type in ['HF', 'VF']:
                if A0_fix is not None:
                    A0_min = A0_fix - 1.
                    A0_max = A0_fix + 1.
                else:
                    # A0 should not be larger than peak between 1.1 and 1.8 Hz
                    A0_max = np.max(10 * np.log10(
                        p_sig[np.array((f > 1.1, f < 1.8)).all(axis=0)])) + 6.
                    A0_min = -240.
                # tstar must be so large than event is below noise
                if max(f[bol_fitting_mask]) < 4:
                    ifreq = np.array((f > 6.0, f < 7.0)).all(axis=0)
                    tstar_min = (np.log(10 ** (A0_max / 10)) -
                                 np.log(np.mean(p_sig[ifreq]))) \
                                / (np.pi * 6.5)
                else:
                    tstar_min = 0.02
                try:
                    A0, f_24, f_c, tstar, width_24, ampfac = fit_peak_att(
                        f=f[bol_fitting_mask],
                        p=p_sig[bol_fitting_mask],
                        A0_max=A0_max,
                        tstar_min=tstar_min,
                        A0_min=A0_min)
                    if A0_fix is None:
                        A0_err = 5.
                    else:
                        A0_err = 0.
                        A0 = A0_fix

                except RuntimeError:
                    pass

            # Fitting LF family events
            else:
                # res = np.polyfit(f[bol_1Hz_mask],
                #                  10 * np.log10(p_sig[bol_1Hz_mask]),
                #                  deg=1)
                res = linregression(x=f[bol_fitting_mask],
                                    y=10 * np.log10(p_sig[bol_fitting_mask]),
                                    q=0.95)
                # A0 = res[1]
                # tstar = - res[0] / 10. * np.log(10) / np.pi  # Because dB
                A0 = res[0]
                tstar = - res[2] / 10. * np.log(10) / np.pi  # Because dB
                A0_err = res[1]
                tstar_err = - res[3] / 10. * np.log(10) / np.pi  # Because dB

    if event_type not in ['LF', 'BB']:
        bol_24_mask = np.array((f > mute_24[0],
                                f < mute_24[1],
                                p_sig > p_noise)).all(axis=0)
        p_red = p_sig - p_noise
        A_24_red, f_24_red, tmp = fit_peak(f[bol_24_mask], p_red[bol_24_mask])
        A_24, f_24, tmp = fit_peak(f[bol_24_mask], p_sig[bol_24_mask])
        # A_24_noise, f_24_noise, tmp_noise = fit_peak(f[bol_24_mask], p_noise[bol_24_mask])
        if width_24 is None:
            width_24 = tmp

    amps = dict()
    amps['A0'] = A0
    amps['tstar'] = tstar
    amps['A0_err'] = A0_err
    amps['tstar_err'] = tstar_err
    amps['A_24'] = A_24
    amps['f_24'] = f_24
    amps['A_24_red'] = A_24_red
    amps['f_24_red'] = f_24_red
    amps['f_c'] = f_c
    amps['width_24'] = width_24
    amps['ampfac'] = ampfac
    amps['fitting_mask'] = bol_fitting_mask
    return amps


def fit_spectra_modes(f_sig, p_sig, mute_24, fminmax, width_peak, ampFactor, kind=None):
    import matplotlib.pyplot as plt

    f = f_sig

    mute_24 = mute_24

    if np.isnan(p_sig[0]):
        f_mode = np.nan
        A_peak = np.nan
        A_baseline = np.nan
        width_mode = np.nan
        ampfac_mode = np.nan

    else:
        width_mode = None
        f_mode = None
        A_baseline = None

        bol_24_mask = np.array((f > mute_24[0],
                                f < mute_24[1])).all(axis=0)



        A_baseline, f_mode, width_mode, ampfac_mode = fit_peak_modes(f[bol_24_mask], p_sig[bol_24_mask],
                                        f0_min = fminmax[0], f0_max = fminmax[-1],
                                        fw_min = width_peak[0], fw_max = width_peak[-1],
                                        ampfac_min = ampFactor[0], ampfac_max=ampFactor[-1]) #50,550 works

        #test if curve fitting was successful or not
        if ampfac_mode: 
            # A_peak = A_baseline+10*np.log10(1+ampfac_mode) #wrong?
            fitted_curve = lorentz_modes(x=f[bol_24_mask],A=A_baseline, x0=f_mode, xw=width_mode, ampfac=ampfac_mode)
            A_peak = max(fitted_curve)
        elif not ampfac_mode:
            A_peak = np.nan


        #debug plot part1
        plt.plot(f, p_sig)
        plt.plot(f[bol_24_mask], p_sig[bol_24_mask])
        # debug plot part2
        plt.plot(f[bol_24_mask],lorentz_modes(x=f[bol_24_mask],A=A_baseline, x0=f_mode, xw=width_mode, ampfac=ampfac_mode))
        plt.plot(f,lorentz_modes(x=f,A=A_baseline, x0=f_mode, xw=width_mode, ampfac=ampfac_mode))
        # plt.axhline(y=A_peak)
        # plt.axhline(y=A_baseline)
        # plt.axvline(x=f_mode)
        # plt.axvline(x=fminmax[0])
        # plt.axvline(x=fminmax[-1])
        # plt.ylim(-250,-150)
        plt.text(x=1, y=-220, s=f'baseline: {A_baseline:6.1f}dB f: {f_mode:6.3f}Hz \n width: {width_mode:6.4f}Hz ampfac: {ampfac_mode:6.1f} \n Peak:{A_peak}dB')
        plt.show()

        if kind == 'spectogram':
            if ((ampfac_mode  > 380.0 and A_peak < -188) or ampfac_mode  < 1.0 or (ampfac_mode  > 580.0 and width_mode > 0.3)): 
                f_mode = np.nan
                A_peak = np.nan
                A_baseline = np.nan
                width_mode = np.nan
                ampfac_mode = np.nan
        elif kind == 'scalogram':
            if ((ampfac_mode  > 500.0 and width_mode > 0.3) or (ampfac_mode  < 1.0)): 
                f_mode = np.nan
                A_peak = np.nan
                A_baseline = np.nan
                width_mode = np.nan
                ampfac_mode = np.nan


    amps = dict()
    amps['A_baseline'] = A_baseline
    amps['A_peak'] = A_peak
    amps['f_mode'] = f_mode
    amps['width_mode'] = width_mode
    amps['ampfac'] = ampfac_mode
    return amps
