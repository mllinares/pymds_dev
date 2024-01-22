#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:47:19 2023

@author: llinares
"""

import numpy as np
import geometric_scaling_factors
# import forward_function
from constants import constants
import torch
# import numpy as np
from math import pi, sin
from chemistry_scaling import clrock, clcoll


def add_noise(input_array, name_file, noise_level, write_data_file, data_file):
    
    """ Add noise to specific dataset.
    INPUTS : input_array, smooth dataset, type : numpy array
             name_file, name of ouput file, type : str
             noise_level, level of noise, type : float
             write_data_file, write cl36 datafile, type : bool
            
    OUTPUTS : "noised" array, type : numpy array
              .txt file containing the array with noise """
    
    noise = np.array(np.random.randn(len(input_array))) * noise_level # create random noise
    noisy_data = input_array+noise # add noise to your data
    np.savetxt(name_file+str(noise_level)+'.txt', noisy_data) # saving


    if write_data_file == True:
        data_out = data_file.copy()
        data_out[:, 64] = noisy_data
        np.savetxt('data_out.csv', data_out, delimiter=',')
    
    return noisy_data

def create_blank_file(number_of_samples=90, regular_spacing=True, spacing=10):
    """ Generate a blank datafile to create a synthetic profile
    INPUT : seismic_scenario, a seismic scenario, dict
            scaling_factors, geometric scaling factors (from gscale.py), dict
            number_of_samples, number of 36Cl samples, int (default 90)
            regular_spacing, is the space between samples regular, bool (default True)
            spacing, spasing between 36 Cl samples, float, (default 50)
            
    OUTPUT : synthetic_data_file, 2D array shape((number_of_samples, 66))
             saving of blank, .csv (delim=',') """
             
    synthetic_data_file=np.zeros((number_of_samples, 66))
    if regular_spacing==True:
        h_samples=np.linspace(0, spacing*(number_of_samples+1), number_of_samples)
    else:
        random_spacing=np.array(np.random.randn(number_of_samples)) * 0.0001
        h_samples=np.linspace(0, spacing, number_of_samples) + random_spacing
    synthetic_data_file[:, 57] = 1 # to avoid nan values
    synthetic_data_file[:, 60] = 1 # to avoid nan values
    synthetic_data_file[:, 61] = 38*1e5 # to avoid nan values
    synthetic_data_file[:, 62] = h_samples # height of samples
    synthetic_data_file[:, 65] = 1e5 # incertitude on measured [36Cl] 
    np.savetxt('blank.csv', synthetic_data_file, delimiter=',')
    return
    
def gen_synthetic_data(seismic_scenario, blank_datafile):
     
    """ Generate a synthetic datafile
    INPUT : seismic_scenario, a seismic scenario, dict
            scaling_factors, geometric scaling factors (from gscale.py), dict
            number_of_samples, number of 36Cl samples, int (default 90)
            regular_spacing, is the space between samples regular, bool (default True)
            spacing, spasing between 36 Cl samples, float, (default 50)
            
    OUTPUT : synthetic_data_file, 2D array shape((number_of_samples, 66))
             saving of synthetic_data_file, .csv (delim=',') """
             
    import forward_function
    import parameters
    from constants import constants
    
    param = parameters.param()
    param.data = np.loadtxt('blank.csv', delimiter=',') # samples chemestry
    
    """ First calculate scaling factors """
    scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, 3)


    synthetic_profile=forward_function.mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200)
    blank_datafile[:, 64] = synthetic_profile
    blank_datafile[:, 65] = min(blank_datafile[:,64])*0.1
    np.savetxt('synthetic_file.csv',blank_datafile, delimiter=',')
    return synthetic_profile


# Seismic scenario stored in a dict
seismic_scenario={}
seismic_scenario['ages'] = torch.tensor([7000, 2500, 500]) # exhumation ages, older to younger (yr)
seismic_scenario['slips'] = torch.tensor([300, 300, 600]) # slip corresponding to the events (cm)
seismic_scenario['SR'] = 0.3 # long term slip rate of your fault (mm/yr)
seismic_scenario['preexp'] = 5*1e3 # Pre-expositionn period (yr)
seismic_scenario['start_depth'] = seismic_scenario['preexp'] * seismic_scenario['SR'] * 1e-1 # (cm) along the fault plane
seismic_scenario['quiescence'] = 0*1e3 # Quiescence period (yr), must be older than last event

# Handling of quiescence period
if seismic_scenario['quiescence'] !=0 :
    seismic_scenario['ages'] = np.hstack((seismic_scenario['quiescence'] + seismic_scenario['ages'][0], seismic_scenario['ages']))
    seismic_scenario['slips'] = np.hstack((0, seismic_scenario['slips']))

# Start creating synthetic datafile
create_blank_file()
data_blank=np.loadtxt('blank.csv', delimiter=',')
synth=gen_synthetic_data(seismic_scenario, data_blank)