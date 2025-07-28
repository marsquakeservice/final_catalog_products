# -*- coding: utf-8 -*-
import json
import logging

class FittingParameterPool:
    COMPONENT_Z = 'Z'
    COMPONENT_N = 'N'
    COMPONENT_E = 'E'
    COMPONENT_R = 'R'
    COMPONENT_T = 'T'

    # Some small info from the GUI. 
    INFO = 'Event info'
    COMP_INFO = "Component info"
    LAST_MODIFIED = 'last modified'

    # Noise and signal windows, also controlled from the GUI
    PICK_NOISE_START = 'noise_start'
    PICK_NOISE_END = 'noise_end'
    PICK_P_START = 'P_spectral_start'
    PICK_P_END = 'P_spectral_end'
    PICK_S_START = 'S_spectral_start'
    PICK_S_END = 'S_spectral_end'

    # Fit parameters, controlled from GUI
    SPECTRA_CORNER_FREQ = 'cornerfrequency'
    SPECTRA_AMPL = 'amplification'
    SPECTRA_A0 = 'A0'
    SPECTRA_StoP = 'StoPratio'
    SPECTRA_TSTAR = 'tstar'
    SPECTRA_WIDTH = 'spectralwidth'
    SPECTRA_F0 = 'f0'
    SPECTRA_FMAX_P = "fmaxP"
    SPECTRA_FMAX_S = "fmaxS"
    SPECTRA_FMIN_P = "fminP"
    SPECTRA_FMIN_S = "fminS"
    SPECTRA_SLOPE = "slope"
    SPECTRA_TSTAR_LOW = "tstar-low"
    SPECTRA_TSTAR_HIGH = "tstar-high"
    SPECTRA_CORNER_FREQ_LOW = "cornerfreq-low"
    SPECTRA_CORNER_FREQ_HIGH = "cornerfreq-high"
    SPECTRA_A0_LOW = "A0-low"
    SPECTRA_A0_HIGH = "A0-high"
    SPECTRA_StoP_LOW = "StoP-low"
    SPECTRA_StoP_HIGH = "StoP-high"
    SPECTRA_OMEGA = "omega"

    # Other parameters, derived from computations. These should 
    # not trigger updates to the GUI 
    FITTING_A0_HIGH = "A0_high"
    FITTING_A0_LOW = "A0_low"
    FITTING_DISTANCE = "distance"
    FITTING_FIT_FC = "fit_fc"
    FITTING_FIT_FC_SIGMA = "fit_fc_sigma"
    FITTING_FIT_LOG_TSTAR = "fit_log_tstar"
    FITTING_FIT_LOG_TSTAR_SIGMA = "fit_log_tstar_sigma"
    FITTING_FMAX = "fmax"
    FITTING_FMIN = "fmin"
    

    def __init__(self, event_name=None):
        # Store the event name we are working on
        self.event_name = event_name

        # Which component are we working on?
        self.selected_component = self.COMPONENT_Z

        # Fitting parameters per component is stored as a dict
        # as [event name][component][an attribute from this class]
        self.fitting_param = {
            self.INFO: None, 
            self.LAST_MODIFIED: None,
            # Time windows used for fitting and spectra
            self.PICK_NOISE_START: None,
            self.PICK_NOISE_END: None,
            self.PICK_P_START: None,
            self.PICK_P_END: None,
            self.PICK_S_START: None,
            self.PICK_S_END: None,
            # Fitting parameters
            self.SPECTRA_CORNER_FREQ: None,
            self.SPECTRA_AMPL: None,
            self.SPECTRA_A0: None,
            self.SPECTRA_StoP: None,
            self.SPECTRA_TSTAR: None,
            self.SPECTRA_WIDTH: None,
            self.SPECTRA_F0: None,
            self.SPECTRA_SLOPE: None,
            self.SPECTRA_OMEGA: None,
            # Ranges for fitting uncertainty
            self.SPECTRA_TSTAR_LOW: None,
            self.SPECTRA_TSTAR_HIGH: None,
            self.SPECTRA_CORNER_FREQ_LOW: None,
            self.SPECTRA_CORNER_FREQ_HIGH: None,
            self.SPECTRA_A0_LOW: None,
            self.SPECTRA_A0_HIGH: None,
            self.SPECTRA_StoP_LOW: None,
            self.SPECTRA_StoP_HIGH: None,
            }

        # Only spectral ranges are stored at the component level
        for component in self.components():
            self.fitting_param[component] = {
                self.LAST_MODIFIED: None,
                # Frequency ranges for P- and S-phases
                self.SPECTRA_FMAX_P: None,
                self.SPECTRA_FMIN_P: None,
                self.SPECTRA_FMAX_S: None,
                self.SPECTRA_FMIN_S: None}   

    def _createPool(self, dict):
        """ Create a new pool from a dictionary """
        _pool = FittingParameterPool()
        _pool.set_parameters(dict)
        return _pool

    def _get_component_level_keys(self):
        """ 
        Returns the keys that must be stored at the component level.
        Checking against this makes sure parameters are assigned to the
        correct levels (event or a component) in case a mistake is done
        while coding.
        """
        return [self.LAST_MODIFIED,
                self.SPECTRA_FMAX_P,
                self.SPECTRA_FMIN_P,
                self.SPECTRA_FMAX_S,
                self.SPECTRA_FMIN_S]
    
    def _get_event_level_keys(self):
        """ 
        Returns the keys that must be stored at the event level.
        Checking against this makes sure parameters are assigned to the
        correct levels (event or a component) in case a mistake is done
        while coding.
        """
        return [self.INFO, 
                self.LAST_MODIFIED,
                self.PICK_NOISE_START,
                self.PICK_NOISE_END,
                self.PICK_P_START,
                self.PICK_P_END,
                self.PICK_S_START,
                self.PICK_S_END,
                self.SPECTRA_CORNER_FREQ,
                self.SPECTRA_AMPL,
                self.SPECTRA_A0,
                self.SPECTRA_StoP,
                self.SPECTRA_TSTAR,
                self.SPECTRA_WIDTH,
                self.SPECTRA_F0,
                self.SPECTRA_SLOPE,
                self.SPECTRA_OMEGA,
                self.SPECTRA_TSTAR_LOW,
                self.SPECTRA_TSTAR_HIGH,
                self.SPECTRA_CORNER_FREQ_LOW,
                self.SPECTRA_CORNER_FREQ_HIGH,
                self.SPECTRA_A0_LOW,
                self.SPECTRA_A0_HIGH,
                self.SPECTRA_StoP_LOW,
                self.SPECTRA_StoP_HIGH]
    
    def get_selected_component(self):
        return self.selected_component
    
    def to_json(self, include_name):
        if include_name:
            return {self.event_name: self.fitting_param}
        else:
            return self.fitting_param

    def set_value(self, component, key, value):
        if component not in [None] + self.components():
            raise ValueError('Unknown component is given')

        # Make sure noise windows and fitting parameters are stored at 
        # the event level, regardless of the component that was passed 
        # by mistake. LAST_MODIFIED is the only exception.
        if key in self._get_event_level_keys() and key != self.LAST_MODIFIED:
            component = None

        # If a component is given, add the key at the component level. 
        # Otherwise, use the event level. 
        if component is not None:
            self.fitting_param[component][key] = value
        else:
            self.fitting_param[key] = value
        
    def setValue(self, component, key, value):
        """ Same as set_value, to avoid running interruptions when 
        coding in Qt style """
        self.set_value(component, key, value)

    def getValue(self, component, key):
        """ Same as get_value, to avoid running interruptions when 
        coding in Qt style """
        return self.get_value(component=component, key=key)
    
    def get_value(self, component, key):
        if component not in [None] + self.components():
            raise ValueError('Unknown component is given')

        # Make sure noise windows and fitting parameters are taken from 
        # the event level, regardless of the component that was passed 
        # by mistake.
        if key in self._get_event_level_keys():
            component = None

        # If a component is given, look for the key at the component level. 
        # Otherwise, try the event level. 
        if component is not None:
            return self.fitting_param[component][key]
        else:
            return self.fitting_param[key]
        
    def get_parameters(self):
        return self.fitting_param
    
    def round_parameters(self, decimals=5):
        """ Round all parameters to the given number of decimals """
        for key, value in self.fitting_param.items():
            if isinstance(value, float) and key in self.keys_for_diff():
                self.fitting_param[key] = round(value, decimals)
            elif isinstance(value, dict):
                for c_key, c_value in value.items():
                    if isinstance(c_value, float):
                        self.fitting_param[key][c_key] = round(c_value, decimals)
        return self #.get_parameters()
    
    def set_parameters(self, new_parameters):
        for key, value in new_parameters.items():
            if isinstance(value, float) and key in self.keys_for_diff():
                value = round(value, 5) 

            self.set_value(component=None, key=key, value=value)

            for component in self.components():
                if component in new_parameters:
                    for c_key, c_value in new_parameters[component].items():
                        if key in self._get_component_level_keys():
                            if isinstance(c_value, float):
                                c_value = round(c_value, 5)
                            
                            self.set_value(component=component, 
                                           key=c_key, 
                                           value=c_value)
    
    def components(self):
        return [self.COMPONENT_Z, self.COMPONENT_N, self.COMPONENT_E, self.COMPONENT_R, self.COMPONENT_T]
    
    def keys_for_diff(self):
        """ Keys that should trigger GUI updates"""
        return [self.INFO, self.LAST_MODIFIED, self.PICK_NOISE_START, 
                self.PICK_NOISE_END, self.PICK_P_START, self.PICK_P_END, 
                self.PICK_S_START, self.PICK_S_END, self.SPECTRA_CORNER_FREQ, 
                self.SPECTRA_AMPL, self.SPECTRA_A0, self.SPECTRA_StoP,
                self.SPECTRA_TSTAR, self.SPECTRA_WIDTH, self.SPECTRA_F0, 
                self.SPECTRA_FMAX_S, self.SPECTRA_FMAX_P, self.SPECTRA_FMIN_S,
                self.SPECTRA_FMIN_P, self.SPECTRA_SLOPE, self.SPECTRA_OMEGA,
                self.SPECTRA_TSTAR_LOW, self.SPECTRA_TSTAR_HIGH, 
                self.SPECTRA_CORNER_FREQ_LOW, self.SPECTRA_CORNER_FREQ_HIGH, 
                self.SPECTRA_A0_LOW, self.SPECTRA_A0_HIGH, self.SPECTRA_StoP_LOW, 
                self.SPECTRA_StoP_HIGH]
    
    def __eq__(self, other):
        if isinstance(other, FittingParameterPool):
            my_keys = self.keys_for_diff()

            # Check event level for all keys
            for key in my_keys:
                if key in self._get_event_level_keys() and \
                    self.get_value(component=None, key=key) != \
                    other.get_value(component=None, key=key):
                    
                    logging.getLogger().critical(
                        "Event level mismatch: {} {} {} is {} on the other".format(
                            None, key, self.get_value(component=None, key=key),
                            other.get_value(component=None, key=key)))
                    
                    return False
            
            # Check component level for all keys
            for component in self.components():
                for key in my_keys:
                    if key in self._get_component_level_keys() and \
                        self.get_value(component=component, key=key) != \
                        other.get_value(component=component, key=key):
                        
                        logging.getLogger().critical(
                            "Component level mismatch: {} {} {} is {} on the other".format(
                                component, key, self.get_value(component=component, key=key),
                                other.get_value(component=component, key=key)))
                        
                        return False
                    
            logging.getLogger().debug("Two parameter sets are equal")

            return True
        

    def __hash__(self):
        return hash(self.fitting_param)

    def __str__(self) -> str:
        return json.dumps(
            self.fitting_param, indent=4, ensure_ascii=True)
    