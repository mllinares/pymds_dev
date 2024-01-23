# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:10:42 2024

@author: maure
"""

#%% Importing libraries
import forward_function as forward
import geometric_scaling_factors
from datetime import datetime
from constants import constants
import torch
import pyro
import numpy as np
import pyro.distributions as dist
import time
import post_process as fig
import sys
from seismic_scenario import seismic_scenario as true_scenario
from pyro.infer import MCMC, NUTS
import parameters

#%% MCMC parametrization
""" Input seismic scenario """
inferred_scenario={} # open new dict to create seismic scenario
number_of_events = 6 # enter the number of event
true_scenario_known = True # true scenario known? (for plotting purpose)

""" Input parameters"""
param=parameters.param()
cl36AMS = param.cl36AMS
height = param.h
Hfinal = param.Hfinal
sigAMS = np.zeros(len(cl36AMS))+(np.min(cl36AMS)*0.1)
Data = torch.tensor(cl36AMS)
var_cl36=np.var(cl36AMS)

"""Parameters inverted """
invert_slips = True # invert slip array ?
use_rpt = False # use rupture package to find slips
invert_sr = True # invert slip rate ?
invert_quies = False # invert quiescence

""" MCMC parameters used in invert.py """
tic=time.time()
pyro.set_rng_seed(20)
w_step = 10  # number of warmup (~30% of total models)
nb_sample = 40 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.7 # target acceptancy probability (<1)

#%% importing result files
all_36cl_models=np.loadtxt('inferred_cl36.txt')
all_sigma=np.loadtxt('sigma.txt')
all_RMSw=np.loadtxt('RMSw_infered_models.txt')


all_age=np.loadtxt('age.txt')
median_age=np.zeros((number_of_events))
for i in range (0, np.shape(all_age)[1]):
    median_age[i]=np.median(all_age[:,i])
inferred_scenario['ages']=torch.tensor(median_age)

if invert_slips==True:
    all_slip_corrected=np.loadtxt('slip.txt')
    median_slip_corrected=np.zeros((number_of_events))
    for i in range (0, np.shape(all_slip_corrected)[1]):
        median_slip_corrected[i]=np.median(all_slip_corrected[:,i])
    inferred_scenario['slips']=torch.tensor(median_slip_corrected)
    slip_hlines=np.zeros((number_of_events))
    for i in range (0, len(slip_hlines)):
        slip_hlines[i]=Hfinal-np.sum(np.median(all_slip_corrected[0:i]))
elif invert_slips==False and use_rpt==True:
        slips=np.loadtxt('slip_rpt.txt')
        inferred_scenario['slips']=torch.tensor(median_slip_corrected)
        slip_hlines=np.zeros((number_of_events))
        for i in range (0, len(slip_hlines)):
            slip_hlines[i]=Hfinal-np.sum(np.median(all_slip_corrected[0:i]))
else:
    inferred_scenario['slips']=true_scenario['slips']
    
if invert_sr==True:
    all_sr=np.loadtxt('SR.txt')
    median_SR=np.median(all_sr)
    inferred_scenario['SR']=median_SR
else:
    inferred_scenario['SR']=true_scenario['SR']
    
if invert_quies==True:
    all_quies=np.loadtxt('quies.txt')
    median_quies=np.median(all_quies)
    inferred_scenario['quiescence']=median_quies
else:
    inferred_scenario['quiescence']=true_scenario['quiescence']

inferred_scenario['preexp']=true_scenario['preexp']


# Compute 36Cl for the median model
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, number_of_events+1)
cl_36_inferred=forward.mds_torch(inferred_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200, find_slip = invert_slips)
#%% Plotting 
tic_pp=time.time()
true_age=true_scenario['ages']
true_slips=true_scenario['slips']

""" Plot 36Cl profiles:
    1) median model with plot profile
    2) median model and greyfill between all models
    3) median model and greyfill between the last 70% of all models (when inversion is stable)"""
    
fig.plot_profile(cl36AMS, param.sig_cl36AMS, height, cl_36_inferred, 'plot_median')
fig.plot_min_max(cl36AMS, height, all_36cl_models, sigAMS=np.zeros((len(cl36AMS)))+np.min(cl36AMS)*0.1, slips=slip_hlines*1e-2, plot_name='all_models')
fig.plot_min_max(cl36AMS, height, all_36cl_models[:,int(0.3*nb_sample)::], sigAMS=np.zeros((len(cl36AMS)))+np.min(cl36AMS)*0.1, slips=slip_hlines*1e-2, plot_name='models_70_percent')

# plot sigma 
fig.plot_variable_np(all_sigma, 'Sigma', 'Sigma')

# Plot ages infered through time
if true_scenario_known == False or number_of_events!=len(true_age):
    for i in range (0, number_of_events):
        fig.plot_variable_np(all_age[:, i], 'Event '+str(i+1), 'Age', num_fig=i+1) 
else:
    for i in range (0, number_of_events):
        fig.plot_variable_np(all_age[:, i], 'Event '+str(i+1), 'Age', true_value=true_age[i], num_fig=i+1) #true_value=true_age[i],

# Plot slip throug time and 2D plots
if use_rpt==False and invert_slips==True:
    if true_scenario_known == False and number_of_events!=len(true_slips):
        for i in range (0, number_of_events):
            fig.plot_variable_np(all_slip_corrected[:, i], title='Slip '+str(i+1), var_name='Slip', num_fig=i+1) 
            fig.plot_2D(all_age[:,i], all_slip_corrected[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), median_values=np.array([median_age[i], median_slip_corrected[i]]))
    elif true_scenario_known == True and number_of_events==len(true_slips):
        for i in range (0, number_of_events):
            fig.plot_2D(all_age[:,i], all_slip_corrected[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), true_values=np.array([true_scenario['ages'][i], true_scenario['slips'][i]]), median_values=np.array([median_age[i], median_slip_corrected[i]]))
            fig.plot_variable_np(all_slip_corrected[:, i],true_value=true_slips[i], title='Slip '+str(i+1), var_name='Slip', num_fig=i+1)
    elif true_scenario_known==True and number_of_events!=len(true_age):
        for i in range (0, number_of_events):
            fig.plot_2D(all_age[:,i], all_slip_corrected[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), median_values=np.array([median_age[i], median_slip_corrected[i]]))
            fig.plot_2D(all_age[:,i], all_slip_corrected[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1)+'_with_true_values', true_values=np.array([true_scenario['ages'].numpy(), true_scenario['slips'].numpy()]), median_values=np.array([median_age[i], median_slip_corrected[i]]))

if invert_sr == True and true_scenario_known==True:
    fig.plot_variable_np(all_sr, 'SR', 'SR (mm/yr)', true_value = true_scenario['SR'])
elif invert_sr == True and true_scenario_known==False:
    fig.plot_variable_np(all_sr, 'SR', 'SR (mm/yr)')

#%%
# if invert_slips==True and true_scenario_known==True and number_of_events==len(true_age):
#     for i in range (0, number_of_events):
#         fig.plot_2D(all_age[:,i], all_slip[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), true_values=np.array([true_scenario['ages'][i], true_scenario['slips'][i]]), median_values=np.array([median_age[i], median_slip_corrected[i]]))
# elif invert_slips==True and true_scenario_known==True and number_of_events!=len(true_age):
#     for i in range (0, number_of_events):
#         fig.plot_2D(all_age[:,i], all_slip[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), median_values=np.array([median_age[i], median_slip_corrected[i]]))
#         fig.plot_2D(all_age[:,i], all_slip[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1)+'_with_true_values', true_values=np.array([true_scenario['ages'].numpy(), true_scenario['slips'].numpy()]), median_values=np.array([median_age[i], median_slip_corrected[i]]))
# elif invert_slips==True and true_scenario_known==False and number_of_events!=len(true_age):
#     for i in range (0, number_of_events):
#         fig.plot_2D(all_age[:,i], all_slip[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), median_values=np.array([median_age[i], median_slip_corrected[i]]))
    
toc_pp=time.time()

print('\nTime for plotting : ', '{0:.2f}'.format((toc_pp-tic_pp)/60), 'min')
