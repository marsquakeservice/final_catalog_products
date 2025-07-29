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

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from magnitudes import lorentz_att, calc_magnitude, lorentz

tstarfac = dict(P=1./3, S=1.)

def ratio_HV():
    # raw_spec = np.asarray([
    #     [0.05 , 1.312], [0.700, 1.190], [1.183, 1.169],
    #     [1.869, 1.162], [2.693, 1.149], [3.500, 1.268],
    #     [4.549, 1.561], [5.307, 2.274], [5.973, 3.002],
    #     [6.519, 4.126], [7.262, 6.593], [7.684, 7.149],
    #     [8.386, 7.026], [9.438, 6.080], [11.068, 6.151],
    #     [12.781, 7.445], [15.618, 7.067], [19.682, 9.222],
    #     [22.267, 15.080], [23.201, 21.092], [24.175, 28.661],
    #     [25.713, 19.907], [26.518, 12.034], [30.000, 8.120]])
    
    this_path = os.path.dirname(os.path.abspath(__file__))
    raw_spec = np.loadtxt(
        os.path.join(this_path, "data", "Carrasco_H_over_V_corrected.csv"), 
            delimiter=',')
    
    return interp1d(x=raw_spec[:, 0], y=raw_spec[:, 1], kind='cubic', 
                    bounds_error=False, fill_value='extrapolate')
        
def real2dB(amplitude):
    return 10 * np.log10(amplitude)

def dB2real(ampfac_dB):
    return 10 ** (0.1 * ampfac_dB)

def hessian2ellipse(hessian, irow, icol):
    mat_red = np.asarray(((hessian[irow, irow], hessian[irow, icol]),
                          (hessian[icol, irow], hessian[icol, icol])))

    w, v = np.linalg.eig(mat_red)
    lenx = 2 * np.sqrt(w[0])
    leny = 2 * np.sqrt(w[1])
    alpha = np.arctan2(v[0][1], v[0][0])

    return lenx, leny, alpha

def misfit_lorentz(vars, f, p_P, p_S, p_noise, sigma_P=10., sigma_S=5.):
    SP_ratio, A0, f0, fc, tstar, fw, ampfac = vars

    misfit = np.sum(
        np.abs(
            real2dB(p_noise + dB2real(
                lorentz_att(f, A0, f0, fc, tstar, fw, ampfac))) - real2dB(p_S))
        / np.sqrt(f), axis=None) / \
             np.sum(sigma_S / np.sqrt(f))

    misfit += np.sum(np.abs(
        real2dB(p_noise + dB2real(
            lorentz_att(f, A0 + SP_ratio, f0, fc, tstar * tstarfac['P'], fw, ampfac))) - real2dB(p_P))
        / np.sqrt(f), axis=None) / \
              np.sum(sigma_P / np.sqrt(f))
    return misfit

def vectorized_lorentz_att(f: np.array,
                A0: float,
                f0: float,
                f_c: np.array,
                tstar: np.array,
                fw: float,
                ampfac: float) -> np.array:
        w = (f - f0) / (fw / 2.)
        stf_amp = 1 / (1 + (f / f_c.flatten()[:,np.newaxis]) ** 2)
        
        term1 = 1 + ampfac / (1 + w ** 2)
        term2 = stf_amp
        term3 = np.exp(-tstar.flatten()[:,np.newaxis] * f * np.pi)

        result = A0 + 20 * np.log10(term1 * term2 * term3)
        return result

def vectorized_misfit_lorentz(vars, f, p_P, p_S, p_noise, sigma_P=10., sigma_S=5.):
    SP_ratio, A0, f0, fc, tstar, fw, ampfac = vars

    lorentz_P = vectorized_lorentz_att(f, A0 + SP_ratio, f0, fc, tstar * tstarfac['P'], fw, ampfac)
    lorentz_S = vectorized_lorentz_att(f, A0, f0, fc, tstar, fw, ampfac)
    P_in_dB = real2dB(p_noise + dB2real(lorentz_P)) - real2dB(p_P)
    S_in_dB = real2dB(p_noise + dB2real(lorentz_S)) - real2dB(p_S)

    misfit_P = np.abs(P_in_dB) / np.sqrt(f)
    misfit_S = np.abs(S_in_dB) / np.sqrt(f)

    misfit = np.sum(misfit_P, axis=1) / np.sum(sigma_P / np.sqrt(f))
    misfit += np.sum(misfit_S, axis=1) / np.sum(sigma_S / np.sqrt(f))
    
    return misfit.reshape(fc.shape)

def calc_cov_lorentz(f, p_P, p_S, p_noise, SP_ratio, A0, f0, fc, tstar, fw, ampfac):
    res = minimize(fun=misfit_lorentz,
                   x0=(SP_ratio, A0, f0, fc, tstar, fw, ampfac),
                   bounds=((-25, 25),  # SP ratio
                           (-220., -140.),  # A0
                           (2.2, 2.5),  # f0
                           (0.0, 1.2),  # f_c
                           (0.01, 1.5),  # tstar
                           (0.0, 0.4),  # fw
                           (0.0, 40.)),  # ampfac
                   args=(f, p_P, p_S, p_noise))
    return res
