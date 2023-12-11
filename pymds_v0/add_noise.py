#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 08:25:17 2023

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