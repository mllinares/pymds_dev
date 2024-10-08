#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:18:11 2023

@author: Maureen Llinares
"""

import forward_function as forward
import geometric_scaling_factors
from datetime import datetime
# import site_parameters
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
import matplotlib.pyplot as plt

sys.stdout = open('summary.txt', 'w')
today = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

""" Input seismic scenario """
seismic_scenario={}
number_of_events = 3

""" Input parameters"""
param=parameters.param()
cl36AMS = param.cl36AMS
height = param.h
Hfinal = param.Hfinal
sigAMS = np.zeros(len(cl36AMS))+(np.min(cl36AMS)*0.1)
Data = torch.tensor(cl36AMS)
var_cl36=np.var(cl36AMS)

""" Chose parameters to invert """
invert_slips = False # invert slip array ?
use_rpt = False # use rupture package to find slips
invert_sr = False # invert slip rate ?
invert_quies = False # invert quiescence

""" MCMC parameters, to be set with CAUTION """
tic=time.time()
pyro.set_rng_seed(20)
w_step = 10 # number of warmup (~30% of total models)
nb_sample = 4000 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.9 # target acceptancy probability (<1)

""" Find slip with rupture package if use_rpt = True"""

if invert_slips == False and use_rpt == True:
    import ruptures as rpt
    algo = rpt.Dynp(model='rank', min_size=5, jump=1).fit(cl36AMS) # l1, l2, normal, rbf, rank
    my_bkps = np.array(algo.predict(n_bkps=number_of_events-1)) # Number of break-ups = nb_ev-1 
    my_bkps[-1]=my_bkps[-1]-1
    my_bkps=np.hstack((0, my_bkps))
    slips = np.zeros((len(my_bkps)-1))
    for i in range (0, len(my_bkps)-1):
        slips[i]=height[my_bkps[i+1]]-(height[my_bkps[i]])

    slips_r = torch.tensor(slips[::-1].copy()) # reverse slip array, copy used to construct torch tensor
    seismic_scenario['slips']=slips_r
    
    

""" Compute geometric scaling factors """
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, number_of_events+1)

""" Defining MCMC model """
def model(obs):
    
    ages = torch.zeros((number_of_events))
    ages[0] = pyro.sample('age1', dist.Uniform(2.0, 30*1e3))
    for i in range (1, number_of_events):
        max_age = ages[i-1]
        ages[i] = pyro.sample('age'+str(i+1), dist.Uniform(2.0, max_age))  
    seismic_scenario['ages'] = ages
    
    if invert_slips == True:
        slips = torch.zeros((number_of_events))
        slips[0]=pyro.sample('slip1', dist.Uniform(1, 400))
        for i in range(1, number_of_events):
            max_slip=Hfinal-torch.sum(slips)
            slips[i] = pyro.sample('slip'+str(i+1), dist.Uniform(0, max_slip))
        seismic_scenario['slips'] = slips
    elif invert_slips == False and use_rpt == False:
        seismic_scenario['slips'] = true_scenario['slips']
        
    if invert_sr==True:
        seismic_scenario['SR'] = pyro.sample('SR', dist.Uniform(0.1, 1))
    else :
        seismic_scenario['SR'] = true_scenario['SR']
    seismic_scenario['preexp'] = true_scenario['preexp']
    seismic_scenario['quiescence'] = true_scenario['quiescence']
    sigma=pyro.sample('sigma', dist.Uniform(0, var_cl36*1e4)) # helps for the inference
    
    t = forward.mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200, find_slip = invert_slips)
    return pyro.sample('obs', dist.Normal(t, sigma), obs=obs)

""" Running MCMC """
kernel = NUTS(model, max_tree_depth = tree_depth, target_accept_prob = target_prob, jit_compile=False) # chose kernel (NUTS, HMC, ...)
mcmc = MCMC(kernel, warmup_steps=w_step, num_samples=nb_sample) 
mcmc.run(obs=Data)
posterior_samples = mcmc.get_samples()
toc=time.time()
print('MCMC done \n start time:', today,'runtime:', '{0:.2f}'.format((toc-tic)/3600), 'hours \n')

""" Plotting and saving with post_process.py """
tic_pp=time.time()

# Saving samples
all_age, median_age, mean_age = fig.array_results(nb_sample, number_of_events, 'age', posterior_samples)
all_sigma, median_sigma, mean_sigma = fig.variable_results(nb_sample,'sigma', posterior_samples)

if invert_slips==True:
    all_slip, median_slip, mean_slip = fig.array_results(nb_sample, number_of_events, 'slip', posterior_samples)
    all_slip_corrected, mean_slip_corrected, median_slip_corrected = fig.correct_slip_amout(all_slip, mean_slip, median_slip, Hfinal) # slip amount correction

if invert_sr==True:
    all_sr, mean_sr, median_sr=fig.variable_results(nb_sample,'SR', posterior_samples)

# Define infered scenario
inferred_scenario={}
inferred_scenario['ages'] = torch.tensor(median_age) 
inferred_scenario['preexp'] = true_scenario['preexp']


if invert_slips == True:
    inferred_scenario['slips'] = torch.tensor(median_slip_corrected)
    slip_hlines=np.zeros((number_of_events))
    for i in range (0, len(slip_hlines)):
        slip_hlines[i]=Hfinal-np.sum(inferred_scenario['slips'].detach().numpy()[0:i])
        
elif invert_slips == False and use_rpt == False:
    inferred_scenario['slips'] = true_scenario['slips']
    slip_hlines=np.zeros((number_of_events))
    for i in range (0, len(slip_hlines)):
        slip_hlines[i]=Hfinal-np.sum(true_scenario['slips'].detach().numpy()[0:i])
else:
    inferred_scenario['slips'] = slips_r
    slip_hlines=np.zeros((number_of_events))
    for i in range (0, len(slip_hlines)):
        slip_hlines[i]=Hfinal-np.sum(inferred_scenario['slips'].detach().numpy()[0:i])
        
if invert_sr == True:
    inferred_scenario['SR'] = median_sr
else:
    inferred_scenario['SR'] = true_scenario['SR']
inferred_scenario['quiescence'] = true_scenario['quiescence']
inferred_scenario['erosion_rate'] = true_scenario['erosion_rate']

# Compute 36Cl for the median model
cl_36_inferred=forward.mds_torch(inferred_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200, find_slip = invert_slips)

# Compute all 36Cl models and save RMSw
all_36cl_models=np.zeros((len(cl36AMS), len(all_age)))
all_RMSw=np.zeros((len(all_age)))

for i in range (0, len(all_age)):
    inferred_scenario['ages'] = torch.tensor(all_age[i])
    if invert_slips == True:
       inferred_scenario['slips'] = torch.tensor(all_slip_corrected[i])
    if invert_sr == True:
        inferred_scenario['SR'] = torch.tensor(all_sr[i])
    else :
        inferred_scenario['SR'] = true_scenario['SR']
    all_36cl_models[:,i]=forward.mds_torch(inferred_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200, find_slip = invert_slips)
    all_RMSw[i]=fig.RMSw(param.cl36AMS, all_36cl_models[:,i], incertitude=sigAMS)
np.savetxt('inferred_cl36.txt', all_36cl_models)
np.savetxt('RMSw_infered_models.txt', all_RMSw)

# Get rhat values
rhat_age=fig.get_rhat_array('age', mcmc.diagnostics(), number_of_events)
if invert_slips == True:
    rhat_slip=fig.get_rhat_array('slip', mcmc.diagnostics(), number_of_events)
if invert_sr == True:
    rhat_sr=fig.get_rhat('SR', mcmc.diagnostics())
rmsw_median_model=fig.RMSw(param.cl36AMS, cl_36_inferred.detach().numpy(), incertitude=sigAMS)

# printing info on the inversion
print('\n SUMMARY', param.site_name, ' \n ', 'warmup : ', w_step, '| samples : ', nb_sample, ' \n tree depth :', tree_depth, ' |target acceptancy :', target_prob, '\n\n  age :', median_age, '\n  rhat_age :', rhat_age)
if invert_slips == True:
    print('\n  slip :', median_slip_corrected, '\n  rhat_slip :', rhat_slip)
if invert_sr == True:
    print('\n  SR :', median_sr, '\n  rhat_SR :', rhat_sr)
if use_rpt==True:
    print('\n  slip :', slips_r)
print('\n  divergences :', len(mcmc.diagnostics()['divergences']['chain 0']))
print('\n  RMSw on median model : ', rmsw_median_model)


# Plotting 
fig.plot_profile(cl36AMS, param.sig_cl36AMS, height, cl_36_inferred, 'plot_name')
fig.plot_min_max(cl36AMS, height, all_36cl_models, sigAMS=np.zeros((len(cl36AMS)))+np.min(cl36AMS)*0.1, slips=slip_hlines*1e-2)
true_age=true_scenario['ages']

for i in range (0, number_of_events):
    fig.plot_variable_np(all_age[:, i], 'Event '+str(i+1), 'Age', true_value=true_age[i], num_fig=i+1)
    
fig.plot_variable_np(all_sigma, 'Sigma', 'Sigma')

if invert_slips == True :
    true_slips=true_scenario['slips']
    for i in range (0, number_of_events):
        fig.plot_variable_np(all_slip_corrected[:, i], title='Slip '+str(i+1), var_name='Slip',true_value=true_slips[i], num_fig=i+1)
if invert_sr == True :
    fig.plot_variable_np(all_sr, 'SR', 'SR (mm/yr)', true_value = true_scenario['SR'])

if invert_slips==True:
    for i in range (0, number_of_events):
        fig.plot_2D(all_age[:,i], all_slip[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), true_values=np.array([true_scenario['ages'][i], true_scenario['slips'][i]]))
toc_pp=time.time()

print('\nTime for plotting : ', '{0:.2f}'.format((toc_pp-tic_pp)/60), 'min')
