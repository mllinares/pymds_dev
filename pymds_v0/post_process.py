#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:01:12 2023

@author: llinares
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_variable(variable, true_value, inferred_value, posterior_sample, w_step, nb_models, num_fig):
    
    """ Figue showing two plots: 1) Evolution of the variable, 2) density plot
        INPUTS : variable, name of the variable, type : string
                 true_value, the true value (when known, set to 'nan' if unknown), type : float
                 inferred_value, inferred value, type : float
                 posterior_sample, tensor of all models (output of MCMC); type : torch tensor
                 w_step, nmber of warm-ups, type : integer 
                 num_fig, number of the plot, type : interger
        OUTPUT : saved plot, PNG format, dpi : 1200 """
        
    # plt.clf()
    plt.figure(num=num_fig, figsize=(10, 5), dpi=1200) # start figure
    plt.suptitle(variable+', true='+str(true_value)) # title
    
    """ 1) Evolution of the variable through the inferrence """
    plt.subplot(1,2,1)
    
    # Horizontal lines to show inferred and true value
    if type(true_value)!=str:
        plt.hlines(true_value, 0, len(posterior_sample[variable]), color='seagreen', label='True value') # plot true value if known  
    # plt.hlines(inferred_value, 0, len(posterior_sample[variable]), color='firebrick', label='Inferred value') # plot mean of inferred value
    plt.hlines(inferred_value, 0, nb_models, color='firebrick', label='Inferred value (median)') # plot mean of inferred value
    plt.plot(posterior_sample[variable], color='black') # plot the variable
    
    # greyfill the warmup portion of the plot
    # lower = np.zeros((w_step))+min(posterior_sample[variable].detach().numpy())-0.1
    # upper = np.zeros((w_step))+max(posterior_sample[variable].detach().numpy())
    # warm = np.arange(0, w_step)
    # plt.fill_between(warm, lower, upper, alpha=0.3)
    
    plt.ylabel(variable)
    plt.xlabel('Iteration')
    plt.legend(loc='upper right')
    
    """ 2) density plot"""
    plt.subplot(1,2,2)
    plt.hist(posterior_sample[variable], bins=10, color='silver') # Density plot of all models
    plt.hist(posterior_sample[variable][0:w_step], bins=10, alpha=0.3) # Density plot of warmup
    plt.hist(posterior_sample[variable][w_step::], bins=10, color='black') # Density plot of models without warmup
    
    # Vertical lines of true and inferred values
    if type(true_value)!=str:
        plt.vlines(true_value, 0, nb_models, color='seagreen', label='True value') # plot true value if known  
    plt.vlines(inferred_value, 0, nb_models, color='firebrick', label='Inferred value') # plot mean of inferred value
    
    plt.ylabel('Density of models')
    plt.xlabel(variable)
    plt.legend(loc='upper right')
    
    plt.savefig(variable+'.png', dpi=1200)
    
def plot_profile(clAMS, sigAMS, height, inferred_cl36, plot_name, num_fig):
    
    """ Plot resulting cl36 profile
    INPUTS : clAMS, measured 36Cl (at/g), type : array
             sigAMS, incertitude (if unknown set to 'nan'), type : array
             height, heigth of samples (m), type : array
             inferred_cl36, inferred 36cl, type : array
             plot_name : name of the plot, type : str
    OUTPUTS : PNG plot """
    plt.figure(num=num_fig)
    if type(sigAMS)!=str:
        plt.errorbar(clAMS, height*1e-2, sigAMS, marker='.', linestyle='', color='black', label='Measured concentration')
    else:
        plt.plot(clAMS, height*1e-2, marker='.', linestyle='', color='black', label='Measured concentration')
    plt.plot(inferred_cl36, height*1e-2, color='lightsteelblue', label='Inferred concentration')
    plt.title('Resulting Profile')
    plt.xlabel('[$^{36}$Cl] (at/g)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.savefig(plot_name+'.png', dpi = 1200)