#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:47:19 2023

@author: llinares
"""

import numpy as np

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

def gen_synthetic_data(seismic_scenario, number_of_samples=90, regular_spacing=True, spacing=5):
    synthetic_data_file=np.zeros((number_of_samples, 66))
    if regular_spacing==True:
        h_samples=np.arange(0, number_of_samples, spacing)
        
    return synthetic_data_file