# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:33:19 2024

@author: maure
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm # for the progress bar
import forward_function as forward
import geometric_scaling_factors
from datetime import datetime
from constants import constants
import torch
# import pyro
import numpy as np
# import pyro.distributions as dist
import time
import post_process as fig
import sys
from seismic_scenario import seismic_scenario as true_scenario
# from pyro.infer import MCMC, NUTS
import parameters

#%% Initializing the input parameters
number_of_events = 6 # enter the number of event
true_scenario_known = True # true scenario known? (for plotting purpose)
seismic_scenario={}

param=parameters.param()
cl36AMS = param.cl36AMS
height = param.h
Hfinal = param.Hfinal
trench_depth = param.trench_depth
Hscarp = Hfinal - trench_depth
sigAMS = param.cl36AMS

#%% Iniatializing search grid
possible_ages = np.arange(0, 18*1e3, 1)
possible_slips = np.arange(50, 400, 10)
possible_SR = np.arange(0.1, 6, 0.1)
step = 10000

#%% Sampling inside the grid
np.random.seed(0)
ages=np.zeros((step, number_of_events))
slips=np.zeros((step, number_of_events))
ages[:,0]=np.random.uniform(low=0, high=18*1e3, size=(step,))
SR=np.random.uniform(low=0.1, high=2, size=(step,))

# Order age array
for i in range(0, step):
    for j in range(1, number_of_events):
        ages[i, j]=np.random.uniform(low=0, high=ages[i, j-1])


for j in range(0, number_of_events):
    np.random.seed(j+10)
    slips[:,j]=np.random.uniform(low=0, high=400, size=(step,))

# Make sure sum slips array is not higher than fault scarp     
for i in range (0, len(ages)):
    slips[i,:] = ((slips[i,:]/np.sum(slips[i,:]))*Hscarp)


#%% Initializing output variables
all_models=np.zeros((step, len(cl36AMS)))
all_RMS=np.zeros((step))

#%% Processing scaling factors
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, number_of_events+2)

#%% Processing all models
seismic_scenario['preexp'] = true_scenario['preexp']
seismic_scenario['quiescence'] = true_scenario['quiescence']
for i in tqdm(range(step), desc="Processing models"):
    
    seismic_scenario['ages'] = torch.tensor(ages[i,:])
    seismic_scenario['slips'] = torch.tensor(slips[i,:])
    seismic_scenario['SR'] = SR[i]
    cl_36 = forward.mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200, find_slip = False)
    all_models[i,:] = cl_36
    all_RMS[i]=fig.RMSw(observed_data=cl36AMS, modeled_data=cl_36.detach().numpy(), incertitude=sigAMS)


#%% Find the best model
best_model_index=np.where(all_RMS==np.min(all_RMS))[0]
print(ages[best_model_index])
print(slips[best_model_index])
print(SR[best_model_index])
all_RMS[np.where((all_RMS>all_RMS[best_model_index]*10))[0]]=all_RMS[best_model_index]*10 # Rescale RMS to se best on plot

#%% Ploting
plt.figure(figsize=(int(5*number_of_events),5))
for i in range (0, number_of_events):
    
    plt.subplot(1,3,1)
    plt.scatter(ages[:,0], slips[:,0], c=(all_RMS), cmap='viridis_r')
    plt.plot(ages[best_model_index, 0], slips[best_model_index, 0], marker='*', color='firebrick', markersize=12)
    plt.subplot(1,3,2)
    plt.scatter(ages[:,1], slips[:,1], c=(all_RMS), cmap='viridis_r')
    plt.plot(ages[best_model_index, 1], slips[best_model_index, 1], marker='*', color='firebrick', markersize=12)
    plt.subplot(1,3,3)
    plt.scatter(ages[:,2], slips[:,2], c=(all_RMS), cmap='viridis_r')
    plt.plot(ages[best_model_index, 2], slips[best_model_index, 2], marker='*', color='firebrick', markersize=12)

