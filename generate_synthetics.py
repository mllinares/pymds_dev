#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:47:19 2023

@author: llinares
"""

import numpy as np
import geometric_scaling_factors
import torch
import matplotlib.pyplot as plt
import os


""" Script to generate your synthtetic datafile:
    1) Edit parameter.py file : 
       Required : self.data = np.loadtxt('blank.csv', delimiter=',') # samples chemestry
       Optional : edit other values (angle, density)
    2) Edit the seismic scenario below (CAUTION the sum of slips should be equal to the eight of your scarp)
    3) Run the script
    
    A plot of your synthetic profile is displayed and the synthtetic data file is saved under 'synthetic_file.csv' """

#%% Functions
def add_noise(input_array, noise_level=0.8*1e5, write_data_file=False, data_file=''):
    
    """ Add noise to specific dataset.
    INPUTS : input_array, smooth dataset, type : numpy array
             noise_level, level of noise, default=0.8*1e5, type : float
             write_data_file, write cl36 datafile, default=False, type : bool
             data_file, rock data file, default='', type=np.arra(nb_sample, 66)
            
    OUTPUTS : "noised" array, type : numpy array """
    
    np.random.seed(0)# always get the same random noise
    noise = np.array(np.random.randn(len(input_array))) * noise_level # create random noise
    noisy_data = input_array+noise # add noise to your data
   
    if write_data_file == True:
        data_out = data_file.copy()
        data_out[:, 64] = noisy_data
        np.savetxt('data_out.csv', data_out, delimiter=',')
    
    return noisy_data

def create_blank_file(Hfinal=901, regular_spacing=True, spacing=10):
    """ Generate a blank datafile to create a synthetic profile
    If no parameter is entered, the profile will contain 90 samples with a spacing of 10cm
    
    INPUT : seismic_scenario, a seismic scenario, dict
            scaling_factors, geometric scaling factors (from gscale.py), dict
            number_of_samples, number of 36Cl samples, int (default 90)
            regular_spacing, is the space between samples regular, bool (default True)
            spacing, spasing between 36 Cl samples, float, (default 50)
            
    OUTPUT : synthetic_data_file, 2D array shape((number_of_samples, 66))
             saving of blank, .csv (delim=',') """
    print('blank')
    number_of_samples=len(np.arange(0, Hfinal, spacing))
    synthetic_data_file=np.zeros((number_of_samples, 66))
    if regular_spacing==True:
        h_samples=np.arange(0, Hfinal, spacing)
    else:
        random_spacing=np.random.uniform(low=0.8, high=1.0, size=(number_of_samples,))*10
        h_samples=np.linspace(0, spacing, number_of_samples) + random_spacing
    synthetic_data_file[:, 57] = 1 # to avoid nan values
    synthetic_data_file[:, 60] = 1 # to avoid nan values
    synthetic_data_file[:, 61] = 38*1e5 # to avoid nan values
    synthetic_data_file[:, 62] = h_samples # height of samples
    
    np.savetxt('blank.csv', synthetic_data_file, delimiter=',')
    return
    
def gen_synthetic_data(seismic_scenario, blank_datafile, adding_noise=True, noise_level=0.8*1e5):
     
    """ Generate a synthetic datafile
    INPUT : seismic_scenario, a seismic scenario,type : dict
            blank_datafile, a blank datafile generated with the above function, type : 1D  np.array
            adding_noise, add noise to your synthetic 36cl profile, type : bool
            noise_level, level of noise, default=0.8*1e5, type : float
            
    OUTPUT : synthetic_profile, type : 2D array shape((number_of_samples, 66))
             saving of synthetic_data_file, .csv (delim=',') """
             
    import forward_function
    import parameters
    from constants import constants
    param = parameters.param()
        
    # Scaling factors
    scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, len(seismic_scenario['ages']))
   
    # 36 Cl profile
    synthetic_profile = forward_function.mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200)
    
    # Adding noise
    if adding_noise==True:
        np.savetxt('profile_before_noise.csv', synthetic_profile, delimiter=',')
        synthetic_profile=add_noise(synthetic_profile, noise_level=noise_level)
   
    # Writting and saving to file
    blank_datafile[:, 64] = synthetic_profile
    np.random.seed(0)
    blank_datafile[:, 65] = np.random.uniform(low=0.1, high=1.0, size=(len(blank_datafile[:,64]),)) * min(blank_datafile[:,64])*0.05 # create random incertitude on measurement
    np.savetxt('synthetic_file.csv',blank_datafile, delimiter=',')
    return synthetic_profile


#%% Main script
# Seismic scenario stored in a dict
seismic_scenario={}
seismic_scenario['ages'] = torch.tensor([9000, 4000, 1500]) # exhumation ages, older to younger (yr)
seismic_scenario['slips'] = torch.tensor([300, 300, 300]) # slip corresponding to the events (cm)
seismic_scenario['SR'] = 0.8 # long term slip rate of your fault (mm/yr)
seismic_scenario['preexp'] = 50*1e3 # Pre-expositionn period (yr)
seismic_scenario['start_depth'] = seismic_scenario['preexp'] * seismic_scenario['SR'] * 1e-1 # (cm) along the fault plane
seismic_scenario['quiescence'] = 0*1e3 # Quiescence period (yr), must be older than last event

# Handling of quiescence period
if seismic_scenario['quiescence'] !=0 :
    seismic_scenario['ages'] = np.hstack((seismic_scenario['quiescence'] + seismic_scenario['ages'][0], seismic_scenario['ages']))
    seismic_scenario['slips'] = np.hstack((0, seismic_scenario['slips']))

# Start creating synthetic datafile
create_blank_file(Hfinal=np.sum(seismic_scenario['slips'].detach().numpy()+1)) 
data_blank = np.loadtxt('blank.csv', delimiter=',')
synth = gen_synthetic_data(seismic_scenario, data_blank, noise_level=4*1e5)

#%% Plotting
synthetic_data = np.loadtxt('synthetic_file.csv', delimiter=',')
plt.figure(dpi=1200)
plt.plot(synth, data_blank[:,62], '.')
plt.xlabel ('36Cl [at/g]')
plt.ylabel ('Height (cm)')
plt.errorbar(synthetic_data[:,64], synthetic_data[:, 62], xerr=synthetic_data[:,65], color='black', alpha=0.4, marker='.', linestyle='', label='Synthetic data')
if os.path.isfile('profile_before_noise.csv')==True:
    synthetic_no_noise = np.loadtxt('profile_before_noise.csv', delimiter=',')
    plt.plot(synthetic_no_noise, synthetic_data[:, 62], marker='', linestyle='-', color='orchid', label='Without noise')
plt.legend()