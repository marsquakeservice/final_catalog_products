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
import logging

import datetime as dt

from itertools import product

import numpy as np
import obspy

import catalog
import event

from fittingutils import real2dB, dB2real, lorentz_att, \
                          vectorized_misfit_lorentz, ratio_HV, \
                          calc_cov_lorentz

tstarfac = dict(P=1./3, S=1.)

class Est2DGaussian:
    def __init__(self,
                 x: np.array,
                 y: np.array,
                 value: np.array):
        """
        https://www.astro.rug.nl/~vogelaar/Gaussians2D/2dgaussians.html
        Given the meshgrid (Xg,Yg) in x and y and the image intensities on that grid in I
        then use the moments of your image to calculate constants a, b and c from which we can
        calculate the semi major axis length (1 sigma) and semi minor axis legth (1 sigma) and
        the angle between the major axis and the positive x axis measured counter clockwise.
        If values a, b and c do not comply to the conditions for an ellipse, then return None.
        The calling environment should check the return value."""

        assert len(x.shape) == 2
        assert x.shape == y.shape
        assert x.shape == value.shape
        self.success = False

        M0 = value.sum()
        self.x0 = (x * value).sum() / M0
        self.y0 = (y * value).sum() / M0
        Mxx = (x * x * value).sum() / M0 - self.x0 * self.x0
        Myy = (y * y * value).sum() / M0 - self.y0 * self.y0
        Mxy = (x * y * value).sum() / M0 - self.x0 * self.y0
        D = 2 * (Mxx * Myy - Mxy * Mxy)
        a = Myy / D
        b = Mxx / D
        c = -Mxy / D
        if a * b - c * c < 0 or a <= 0 or b <= 0:
            self.sx = None
            self.sy = None
            self.theta_deg = None
            self.A = None
            self.x0 = None
            self.y0 = None
            self.success = False
            return

        # Find the area of one pixel expressed in grids to find amplitude A.
        # This assumes constant spacing
        nx = x.shape[0]
        ny = x.shape[1]
        dx = abs(x[0, 0] - x[0, -1]) / nx
        dy = abs(y[0, 0] - y[-1, -1]) / ny
        self.A = dx * dy * M0 * (a * b - c * c) ** 0.5 / np.pi

        p = ((a - b) ** 2 + 4 * c * c) ** 0.5
        theta = np.rad2deg(0.5 * np.arctan(2 * c / (a - b)))
        if a - b > 0:  # Not HW1 but the largest axis corresponds to theta.
            theta += 90.0
        if theta < 0:
            theta += 180

        self.theta_deg = theta
        # Major and minor axis lengths
        major = (1 / (a + b - p)) ** 0.5
        minor = (1 / (a + b + p)) ** 0.5
        self.sigma_x = major
        self.sigma_y = minor
        self.success = True

    def eval(self, x, y):  # x, y, A, x0, y0, sigma_x, sigma_y, theta):
        # https://www.astro.rug.nl/~vogelaar/Gaussians2D/2dgaussians.html
        theta = np.radians(self.theta_deg)
        sigx2 = self.sigma_x ** 2
        sigy2 = self.sigma_y ** 2
        a = np.cos(theta) ** 2 / (2 * sigx2) + np.sin(theta) ** 2 / (2 * sigy2)
        b = np.sin(theta) ** 2 / (2 * sigx2) + np.cos(theta) ** 2 / (2 * sigy2)
        c = np.sin(2 * theta) / (4 * sigx2) - np.sin(2 * theta) / (4 * sigy2)

        expo = -a * (x - self.x0) ** 2 - b * (y - self.y0) ** 2 - 2 * c * (x - self.x0) * (y - self.y0)
        return self.A * np.exp(expo)


def timeit(reftime):
    diff = dt.datetime.now() - reftime
    return "{:.2f} s elapsed".format(diff.total_seconds())

class Fitter:
    def __init__(self, catalog, inventory, path_sc3dir):
        self.event = None
        self.values_new = dict()
        self.phase = 'P'
        self.component = 'Z'

        # Load H/V values once when the instance created
        self.HV_ipl = dict()
        self.HV_ipl['Z'] = np.ones_like
        self.HV_ipl['N'] = ratio_HV()
        self.HV_ipl['E'] = ratio_HV()
        self.HV_ipl['R'] = ratio_HV()
        self.HV_ipl['T'] = ratio_HV()
        self.SP_ratio = dict(P=-20, S=0)
        self.SP_ratio_init = dict(P=-20, S=0)
        
        # Dict to store all three components fitting results
        self.three_comp_results = {'Z': None, 'N': None, 'E': None}

        # Initial read
        self.catalog = catalog

        self.path_sc3dir = path_sc3dir
        self.event_names = [ev.name for ev in self.catalog]
        
        logging.getLogger().info("[Fitter] Reading inventory")
        start_time = dt.datetime.now()
        self.inv = inventory
        logging.getLogger().info(
            "[Fitter] Reading inventory...DONE [{}]".format(timeit(start_time)))

        self.event = None
        self.fmin = dict(BB=0.15, XB=0.15, WB=0.15, LF=0.15, HF=0.9, VF=1.1)
        self.fmax = dict(BB=1.5, XB=6.0, WB=6.0, LF=1.5, HF=4.0, VF=6.0)
    
    
    def swap_event(
        self, event_name, detick_nfsamp, instrument, rotate, time_windows=None,
        smprate="", force_products=False):
        """ 
        Change the current event, read its waveforms and calculate spectra 
        
        """
        
        if event_name not in self.get_event_names():
            raise ValueError(
                "event {} is not in the catalog".format(event_name))
        else:
            self.event = self.catalog.select(name=event_name).events[0]
            
            # This would reset event.wf_type to RAW
            #self.event.read_waveforms(inv=self.inv, sc3dir=self.path_sc3dir) 
            
            print("obtaining spectra for event {}, {}/Q{}, wf {}, smprate "\
                "{}, ZRT {}".format(
                    event_name, self.event.mars_event_type_short, 
                    self.event.quality, self.event.wf_type, smprate, rotate))
        
            self.event.calc_spectra(
                winlen_sec=20, detick_nfsamp=detick_nfsamp, 
                time_windows=time_windows, rotate=rotate, instrument=instrument, 
                smprate=smprate, force_products=force_products)
        
        return self.event
    
    
    def get_masked_noise(self):
        f_noise_masked = self.event.spectra['noise']['f']
        np.ma.masked_less(f_noise_masked, value=0.1, copy=False)
        return f_noise_masked
    
    def get_pick(self, label):
        try:
            return self.event.picks[label]
        except:
            return None
        
    def get_available_picks(self):
        """ Returns the available picks for the active event """
        available = []
        for phase in ['P', 'S', 'PP', 'SS', 'Pdiff', 'P1', 'S1', 'Pg', 'Sg', 
                      'x1', 'x2', 'y1', 'y2', 'start', 'end',
                      'noise_start', 'noise_end', 'P_spectral_start',
                      'P_spectral_end', 'S_spectral_start', 'S_spectral_end']:
            if self.get_pick(phase):
                available.append(phase)
        return available
    
    def get_catalog(self):
        return self.catalog
    
    def get_event_type(self):
        return self.event.mars_event_type_short
    
    def get_noise_spectrum(self, component):
        return real2dB(self.event.spectra['noise'][f'p_{component}'])

    def get_noise_spectrum_DL(self, component):
        return real2dB(self.event.spectra_DL['noise'][f'p_{component}'])
    
    def get_noise_spectrum_DG(self, component):
        return real2dB(self.event.spectra_DG['noise'][f'p_{component}'])

    def get_noise_spectrum_SP(self, component):
        return real2dB(self.event.spectra_SP['noise'][f'p_{component}'])
    
    def get_noise_frequency(self):
        f_noise = self.event.spectra['noise']['f']
        return f_noise
    
    def get_noise_frequency_DL(self):
        return self.event.spectra_DL['noise']['f']
    
    def get_noise_frequency_DG(self):
        return self.event.spectra_DG['noise']['f']
    
    def get_noise_frequency_SP(self):
        return self.event.spectra_SP['noise']['f']
    
    def get_phase_frequency(self, phase):
        f_phase = self.event.spectra[phase]['f']
        return f_phase
    
    def get_P_frequency(self):
        return self.get_phase_frequency('P')
    
    def get_P_frequency_DL(self):
        return self.event.spectra_DL['P']['f']
    
    def get_S_frequency_DL(self):
        return self.event.spectra_DL['S']['f']
    
    def get_P_frequency_DG(self):
        return self.event.spectra_DG['P']['f']
    
    def get_S_frequency_DG(self):
        return self.event.spectra_DG['S']['f']
    
    def get_P_frequency_SP(self):
        return self.event.spectra_SP['P']['f']
    
    def get_S_frequency_SP(self):
        return self.event.spectra_SP['S']['f']
    
    def get_S_frequency(self):
        return self.get_phase_frequency('S')
    
    def get_P_spectrum(self, component):
        return real2dB(self.event.spectra['P'][f'p_{component}'])

    def get_S_spectrum(self, component):
        return real2dB(self.event.spectra['S'][f'p_{component}'])
    
    def get_P_spectrum_DL(self, component):
        return real2dB(self.event.spectra_DL['P'][f'p_{component}'])

    def get_S_spectrum_DL(self, component):
        return real2dB(self.event.spectra_DL['S'][f'p_{component}'])
    
    def get_P_spectrum_DG(self, component):
        return real2dB(self.event.spectra_DG['P'][f'p_{component}'])

    def get_S_spectrum_DG(self, component):
        return real2dB(self.event.spectra_DG['S'][f'p_{component}'])

    def get_P_spectrum_SP(self, component):
        return real2dB(self.event.spectra_SP['P'][f'p_{component}'])

    def get_S_spectrum_SP(self, component):
        return real2dB(self.event.spectra_SP['S'][f'p_{component}'])
    
    def get_event_name(self):
        """Return active event's name"""
        return self.event.name
    
    def get_event_names(self):
        """ Returns just the official event names """
        return self.event_names
    
    def get_long_event_names(self):
        """ 
        Returns event names and qualities to be used 
        as extra information; e.g. S0235b [BB, QA]
        """
        names = ["{} [{}, Q{}]".format(
            event.name, event.mars_event_type_short, event.quality) 
            for event in self.catalog.events]
        
        return names
    
    def get_long_name(self, event=None):
        """ 
        Returns a single event's name in the long form, e.g. S0235b [BB, QA]
        """
        if event is None:
            event = self.event

        return "{} [{}, Q{}]".format(
            event.name, event.mars_event_type_short, event.quality)
    
    def get_waveform_info(self):
        low_sps = self.event.waveforms_VBB
        high_sps = self.event.waveforms_SP
        deglitched = self.event.waveforms_DG
        deepl = self.event.waveforms_DL
        tags = ['Low sps', 'High sps', 'Deglitch.', 'Denoised']
        _info = "──────────────────────────────\nWaveform summary:\n"

        for _data, _tag in zip([low_sps, high_sps, deglitched, deepl], tags):
            if _data:
                _info += "├─ {}\t: {}.{}.{} @ {} sps\n".format(_tag,
                    _data[0].stats.station, _data[0].stats.location, 
                    _data[0].stats.channel[0:2], _data[0].stats.sampling_rate)
            else:
                _info += "├─ " + _tag + "\t: Not avaliable\n"
            
        _info += "──────────────────────────────\n"
        return _info
    
    def get_uncertainty(self, component, phase):
        try:
            fmin = self.three_comp_results[component]['uncertainty'][phase]['lowerf']
            fmax = self.three_comp_results[component]['uncertainty'][phase]['upperf']
            ymin = self.three_comp_results[component]['uncertainty'][phase]['lowery']
            ymax = self.three_comp_results[component]['uncertainty'][phase]['uppery']

            return fmin, ymin, fmax, ymax
        except:
            return None, None, None, None
    
    def fit_for_component(self, fitting_parameters, component, tstar_max=2.0):
        
        start = dt.datetime.now()
        f_noise = self.get_noise_frequency()
        f_noise_masked = self.get_masked_noise()

        tstar = fitting_parameters.get_value(component, 'tstar')
        fc = fitting_parameters.get_value(component, 'cornerfrequency')
        
        self.values_new['tstar'] = tstar
        self.values_new['f_c'] = fc
        self.component = component
        self.values_new['A0'] = fitting_parameters.get_value(component, 'A0')
        self.values_new['f0'] = fitting_parameters.get_value(component, 'f0')
        self.values_new['fw'] = fitting_parameters.get_value(component, 'spectralwidth')
        self.values_new['ampfac'] = fitting_parameters.get_value(component, 'amplification')
        self.SP_ratio['P'] = -fitting_parameters.get_value(component, 'StoPratio')
        self.values_new['SP_ratio'] = fitting_parameters.get_value(component, 'StoPratio')
        self.values_new['fminP'] = fitting_parameters.get_value(component, 'fminP')
        self.values_new['fmaxP'] = fitting_parameters.get_value(component, 'fmaxP')
        self.values_new['fminS'] = fitting_parameters.get_value(component, 'fminS')
        self.values_new['fmaxS'] = fitting_parameters.get_value(component, 'fmaxS')
        
        log_tstars = np.linspace(-3., np.log10(tstar_max), 20)
        log_fcs = np.linspace(-1, 1., 20)
        log_ffcs, log_ttstars = np.meshgrid(log_fcs, log_tstars, indexing='ij')
        ttstars = 10. ** log_ttstars
        ffcs = 10. ** log_ffcs

        # The return value as a dict which stores all computational output 
        gamma = fitting_parameters.get_value(None, 'slope')
        omega_exponent = fitting_parameters.get_value(None, 'omega')

        results = {}
        
        # The main manual fitting
        for phase in ['P', 'S']:
            f_phase = self.event.spectra[phase]['f']
        
            y_lorentz_new = lorentz_att(f=f_noise_masked,
                                        A0=self.values_new['A0'] + self.SP_ratio[phase],
                                        f0=self.values_new['f0'],
                                        f_c=self.values_new['f_c'],
                                        tstar=self.values_new['tstar'] * tstarfac[phase],
                                        ampfac=self.values_new['ampfac'],
                                        fw=self.values_new['fw'],
                                        gamma=gamma,
                                        omega_exp=omega_exponent) # + 10 * np.log10(f_noise_masked)

            y_lorentz_new_plot = real2dB(
                dB2real(y_lorentz_new) * self.HV_ipl[component](f_phase) +
                    self.event.spectra['noise'][f'p_{component}'])

            results['y_lorentz_new_' + phase] = y_lorentz_new
            results['y_lorentz_new_plot_' + phase] = y_lorentz_new_plot
            results['f_phase_' + phase] = f_phase
        

        # Fitting uncertainties using the bounds defined by the user
        results['uncertainty'] = {'P': {'lowerf': None, 'upperf': None, 
                                        'lowery': None, 'uppery': None},
                                  'S': {'lowerf': None, 'upperf': None,
                                        'lowery': None, 'uppery': None}}
        
        stop_low = fitting_parameters.get_value(None, 'StoP-low')
        stop_high = fitting_parameters.get_value(None, 'StoP-high')
        tstar_low = fitting_parameters.get_value(None, 'tstar-low')
        tstar_high = fitting_parameters.get_value(None, 'tstar-high')
        fc_low = fitting_parameters.get_value(None, 'cornerfreq-low')
        fc_high = fitting_parameters.get_value(None, 'cornerfreq-high')
        A0_low = fitting_parameters.get_value(None, 'A0-low')
        A0_high = fitting_parameters.get_value(None, 'A0-high')
                
        for phase in ['P', 'S']:
            sum_lower = None
            sum_upper = None
            for _fcu, _tstaru, _A0u, _SPu in product([fc_low, fc_high], 
                                                 [tstar_low, tstar_high],
                                                 [A0_low, A0_high], 
                                                 [stop_low, stop_high]):
                
                f_phase_uncert = self.event.spectra[phase]['f']
                if phase == 'P':
                    _A0 = _A0u - _SPu
                else:
                    _A0 = _A0u

                y_lorentz_new_uncert = lorentz_att(
                    f=f_noise_masked,
                    A0=_A0,
                    f0=self.values_new['f0'],
                    f_c=_fcu,
                    tstar=_tstaru * tstarfac[phase],
                    ampfac=self.values_new['ampfac'],
                    fw=self.values_new['fw'],
                    gamma=gamma, omega_exp=omega_exponent) 
                
                y_lorentz_new_plot_uncert = real2dB(
                    dB2real(y_lorentz_new_uncert) * self.HV_ipl[component](f_phase_uncert)
                    + self.event.spectra['noise'][f'p_{component}'])
                
                if sum_lower:
                    if np.sum(y_lorentz_new_plot_uncert) < sum_lower:
                        sum_lower = np.sum(y_lorentz_new_plot_uncert)
                        results['uncertainty'][phase]['lowerf'] = f_phase_uncert
                        results['uncertainty'][phase]['lowery'] = y_lorentz_new_plot_uncert
                else:
                    sum_lower = np.sum(y_lorentz_new_plot_uncert)
                    results['uncertainty'][phase]['lowerf'] = f_phase_uncert
                    results['uncertainty'][phase]['lowery'] = y_lorentz_new_plot_uncert

                if sum_upper:
                    if np.sum(y_lorentz_new_plot_uncert) > sum_upper:
                        sum_upper = np.sum(y_lorentz_new_plot_uncert)
                        results['uncertainty'][phase]['upperf'] = f_phase_uncert
                        results['uncertainty'][phase]['uppery'] = y_lorentz_new_plot_uncert
                else:
                    sum_upper = np.sum(y_lorentz_new_plot_uncert)
                    results['uncertainty'][phase]['upperf'] = f_phase_uncert
                    results['uncertainty'][phase]['uppery'] = y_lorentz_new_plot_uncert

            
        # Automated fit
        p_P = self.event.spectra['P'][f'p_{component}'] / self.HV_ipl[component](f_noise)
        p_S = self.event.spectra['S'][f'p_{component}'] / self.HV_ipl[component](f_noise)
        
        f_noise = self.event.spectra['noise']['f']
        f_noise_masked = np.ma.masked_outside(x=f_noise,
                                              v1=self.values_new['fminP'],
                                              v2=self.values_new['fmaxP'])
        
        # for i in range(log_ffcs.shape[0]):
        #     for j in range(log_ffcs.shape[1]):
        #         misfits[i, j] = misfit_lorentz2(vars=(
        #             self.SP_ratio['P'],
        #             fitting_parameters.get_value(component, 'A0'),
        #             fitting_parameters.get_value(component, 'f0'),
        #             ffcs[i, j],
        #             ttstars[i, j],
        #             fitting_parameters.get_value(component, 'spectralwidth'), 
        #             fitting_parameters.get_value(component, 'amplification')),
        #             f=f_noise_masked,
        #             p_P=p_P, p_S=p_S,
        #             p_noise=self.event.spectra['noise'][f'p_{component}'])
        misfits = vectorized_misfit_lorentz(
            vars=(self.SP_ratio['P'],
                  fitting_parameters.get_value(component, 'A0'),
                  fitting_parameters.get_value(component, 'f0'),
                  ffcs,
                  ttstars,
                  fitting_parameters.get_value(component, 'spectralwidth'), 
                  fitting_parameters.get_value(component, 'amplification')),
            f=f_noise_masked, p_P=p_P, p_S=p_S,
            p_noise=self.event.spectra['noise'][f'p_{component}'])
        
        prob = np.exp(-0.5 * misfits ** 2.)
        
        # For pcolormesh misfit plot
        results['log_ffcs'] = log_ffcs 
        results['log_ttstars'] = log_ttstars
        results['prob'] = prob
        results['f_noise_masked'] = np.where(
            np.ma.getmask(f_noise_masked), np.nan, f_noise_masked)
        
        # Crosshair for manual fit on pcolormesh
        results['manualfit'] = (np.log10(fc), np.log10(tstar))

        # Best fitting values
        ifc, itstar = np.unravel_index(np.argmax(prob, axis=None), misfits.shape)
        log_fc_best = log_ffcs[ifc, itstar]
        logtstar_best = log_ttstars[ifc, itstar]
        tstar_best = 10 ** logtstar_best
        fc_best = 10 ** log_fc_best
        
        results['bestfit'] = (log_fc_best, logtstar_best)
        results['log_fc_best'] = log_fc_best
        results['logtstar_best'] = logtstar_best
        results['tstar_best'] = tstar_best
        results['fc_best'] = fc_best

        prob_masked = np.ma.masked_where(
            condition=((log_ffcs < log_fc_best - 0.4) |
                       (log_ffcs > log_fc_best + 0.4) |
                       (log_ttstars > logtstar_best + 0.4) |
                       (log_ttstars < logtstar_best - 0.4)),
                       a=prob, copy=True)
            
        gaussfit = Est2DGaussian(x=log_ffcs, y=log_ttstars, value=prob_masked)
        
        if gaussfit.success:
            self.values_new['fit_fc'] = float(gaussfit.x0)
            self.values_new['fit_log_tstar'] = float(gaussfit.y0)
            self.values_new['fit_fc_sigma'] = float(gaussfit.sigma_x)
            self.values_new['fit_log_tstar_sigma'] = float(gaussfit.sigma_y)
            self.values_new['fit_theta'] = float(gaussfit.theta_deg)

            results['fit_fc'] = float(gaussfit.x0)
            results['fit_log_tstar'] = float(gaussfit.y0)
            results['fit_fc_sigma'] = float(gaussfit.sigma_x)
            results['fit_log_tstar_sigma'] = float(gaussfit.sigma_y)
            results['fit_theta'] = float(gaussfit.theta_deg)
        else:
            results['fit_fc'] = None
            results['fit_log_tstar'] = None
            results['fit_fc_sigma'] = None
            results['fit_log_tstar_sigma'] = None
            results['fit_theta'] = None
            
            logging.getLogger().error(
                "[{}] Gaussian fit failed!".format(self.__class__.__name__))
        
        # P-window best fits: Use the P-windows for P waves
        f_noise_maskedP = np.ma.masked_outside(
            x=f_noise, v1=self.values_new['fminP'], 
            v2=self.values_new['fmaxP'])
        
        y_best_fit_P = real2dB(dB2real(
            lorentz_att(f=f_noise_maskedP, 
                        A0=self.values_new['A0'] + self.SP_ratio['P'], 
                        f0=self.values_new['f0'], 
                        f_c=fc_best, 
                        tstar=tstar_best * tstarfac['P'], 
                        fw=self.values_new['fw'],
                        ampfac=self.values_new['ampfac'],
                        omega_exp=omega_exponent)) *
            self.HV_ipl[component](f_noise) +
            self.event.spectra['noise'][f'p_{component}'])
        
        # Best fit for S-wave: Use the S-windows for S waves, not P
        f_noise_maskedS = np.ma.masked_outside(
            x=f_noise, v1=self.values_new['fminS'], 
            v2=self.values_new['fmaxS'])
        
        y_best_fit_S = real2dB(dB2real(
            lorentz_att(f=f_noise_maskedS, A0=self.values_new['A0'], 
                        f0=self.values_new['f0'], f_c=fc_best,
                        tstar=tstar_best, fw=self.values_new['fw'], 
                        ampfac=self.values_new['ampfac'], 
                        omega_exp=omega_exponent)) *
            self.HV_ipl[component](f_noise) +
            self.event.spectra['noise'][f'p_{component}'])
        
        # Store the best fits for P and S. Remove the mask with NaN
        # for further plotting.
        results['y_best_fit_P'] = np.where(
            np.ma.getmask(y_best_fit_P), np.nan, y_best_fit_P)
        results['y_best_fit_S'] = np.where(
            np.ma.getmask(y_best_fit_S), np.nan, y_best_fit_S)
        
        # P and S frequency ranges. Remove the mask with NaN.
        results['f_noise_maskedP'] = np.where(
            np.ma.getmask(f_noise_maskedP), np.nan, f_noise_maskedP)
        results['f_noise_maskedS'] = np.where(
            np.ma.getmask(f_noise_maskedS), np.nan, f_noise_maskedS)

        idx_24 = np.argmin(abs(f_noise_masked - 2.4))
        self.values_new['A_24'] = results['y_lorentz_new_plot_P'][idx_24]
        results['A_24_new'] = results['y_lorentz_new_plot_P'][idx_24]

        SP_f_noise, SP_y_ratio = self.calculate_corrected_fit(
            spectra=self.event.spectra,
            tstar=self.values_new['tstar'],
            component=component,
            ampfac_dB=real2dB(self.values_new['ampfac']),
            f0=self.values_new['f0'],
            fw=self.values_new['fw'])
        
        results['SP_f_noise'] = SP_f_noise
        results['SP_y_ratio'] = SP_y_ratio
        results['distance'] = self.event.distance
        results['A_24'] = self.event.amplitudes['A_24']

        logging.getLogger().info(
            "[{}] Fitting for {} took {} ".format(
                self.__class__.__name__, component, str(timeit(start))))

        self.three_comp_results[component] = results

        return self.three_comp_results
    

    def fit(self, fitting_parameters, tstar_max=2.0):
        """ Fit only the selected component """
        component = fitting_parameters.get_selected_component()
        
        return self.fit_for_component(
            fitting_parameters=fitting_parameters, component=component, 
            tstar_max=tstar_max)
    
    
    def get_fitting_results(self):
        return self.three_comp_results
    
    
    def calculate_corrected_fit(self, spectra, tstar, component, ampfac_dB, f0, fw):
        """
        :param spectra: Spectra dictionary from MQS_reports
        :param tstar: tstar value in seconds
        :param component: component to plot (from spectra object)
        :param ampfac_dB: Amplification factor in dB of 2.4 Hz mode
        :param f0: Center frequency of 2.4 Hz mode
        :param fw: Spectral width of 2.4 Hz mode
        """
        f_noise = spectra['noise']['f']
        p_noise = spectra['noise'][f'p_{component}']
        f = spectra['P']['f']
        p_P = spectra['P'][f'p_{component}']
        p_S = spectra['S'][f'p_{component}']

        Q_term_P = np.exp(-np.pi * f * tstar * 0.25)
        Q_term_S = np.exp(-np.pi * f * tstar)

        y_SP_ratio = real2dB(p_S / Q_term_S ** 2) - real2dB(p_P / Q_term_P ** 2)
        
        return f_noise, y_SP_ratio

