#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:18:11 2023

@author: Maureen Llinares
"""

import forward_function as forward
import geometric_scaling_factors
# import site_parameters
from constants import constants
import torch
import pyro
import numpy as np
import pyro.distributions as dist
import post_process as fig
# from pyro.distributions import constraints
from seismic_scenario import seismic_scenario as true_scenario
from pyro.infer import MCMC, NUTS
import parameters
import matplotlib.pyplot as plt

""" Input seismic scenario """
seismic_scenario={}
erosion_rate = 0 # Erosion rate (mm/yr)
number_of_events = 3
seismic_scenario['erosion_rate'] = erosion_rate

""" Input parameters"""
param=parameters.param()
cl36AMS = param.cl36AMS
height = param.h
Hfinal = param.Hfinal
sigAMS = param.sig_cl36AMS
Data = torch.tensor(cl36AMS)
var_cl36=np.var(cl36AMS)

""" Geometric scaling """
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, number_of_events+1)

""" MCMC parameters """
pyro.set_rng_seed(50)
w_step = 10  # number of warmup (~30% of total models)
nb_sample = 500 # number of samples

""" MCMC model """
def model(obs):
    
    ages = torch.zeros((number_of_events))
    # slips = torch.zeros((number_of_events))
    ages[0] = pyro.sample('age1', dist.Uniform(6*1e3, 50*1e3))

    for i in range (1, number_of_events):
        max_age = ages[i-1]
        ages[i] = pyro.sample('age'+str(i+1), dist.Uniform(2.0, max_age))  
    
    seismic_scenario['ages'] = ages
    seismic_scenario['SR'] = true_scenario['SR']
    seismic_scenario['preexp'] = true_scenario['preexp']
    # infered_slips = true_scenario['slips']
    # slips[0]=pyro.sample('slip1', dist.Uniform(1, 400))
    # for i in range(1, number_of_events):
    #     max_slip=Hfinal-torch.sum(slips)
    #     slips[i] = pyro.sample('slip'+str(i+1), dist.Uniform(0, max_slip))
        
    seismic_scenario['slips'] = true_scenario['slips']
    seismic_scenario['quiescence'] = true_scenario['quiescence']
    sigma=pyro.sample('sigma', dist.Uniform(0, var_cl36*1e4)) # helps for the inference
    
    t = forward.mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200)
    return pyro.sample('obs', dist.Normal(t, sigma), obs=obs)

""" usage MCMC """
kernel = NUTS(model, max_tree_depth = 4, target_accept_prob = 0.9) # chose kernel (NUTS, HMC, ...)
mcmc = MCMC(kernel, warmup_steps=w_step, num_samples=nb_sample) 
mcmc.run(obs=Data)
posterior_samples = mcmc.get_samples()
print('MCMC done \n')
# print(mcmc.diagnostics(), '\n\n')

""" plotting and saving with post_process.py """
# Saving
all_age, median_age, mean_age = fig.array_results(nb_sample, number_of_events, 'age', posterior_samples)
# all_slip, median_slip, mean_slip = fig.array_results(nb_sample, number_of_events, 'slip', posterior_samples)
# all_slip_corrected, mean_slip_corrected, median_slip_corrected = fig.correct_slip_amout(all_slip, mean_slip, median_slip, Hfinal) # slip amount correction

# for i in range (0, number_of_events):
#     fig.plot_variable_np(all_slip_corrected[:, i], title='Slip '+str(i+1), var_name='Slip', num_fig=i+1)

# Import Inferred scenario
inferred_scenario={}
inferred_scenario['ages'] = torch.tensor(median_age) 
inferred_scenario['SR'] = true_scenario['SR']
inferred_scenario['preexp'] = true_scenario['preexp']
inferred_scenario['slips'] = true_scenario['slips']
inferred_scenario['quiescence'] = true_scenario['quiescence']
inferred_scenario['erosion_rate'] = true_scenario['erosion_rate']

# Compute 36Cl for the median model
cl_36_inferred=forward.mds_torch(inferred_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200)

# Compute 
all_36cl_models=np.zeros((len(cl36AMS), len(all_age)))
for i in range (0, len(all_age)):
    inferred_scenario['ages'] = torch.tensor(all_age[i])
    # inferred_scenario['slips'] = torch.tensor(all_slip[i])
    all_36cl_models[:,i]=forward.mds_torch(inferred_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200)
np.savetxt('inferred_cl36.txt', all_36cl_models)

slip_hlines=np.zeros((number_of_events))
for i in range (0, len(slip_hlines)):
    slip_hlines[i]=Hfinal-np.sum(true_scenario['slips'].detach().numpy()[0:i])
    

rhat_age=fig.get_rhat_array('age', mcmc.diagnostics(), number_of_events)
# rhat_slip=fig.get_rhat_array('slip', mcmc.diagnostics(), number_of_events)
rmsw_median_model=fig.RMSw(param.cl36AMS, cl_36_inferred.detach().numpy(), incertitude=sigAMS)

# printing info 
print('\n SUMMARY', param.site_name, ' \n ', 'warmup : ', w_step, ' samples : ', nb_sample, '\n\n  age :', median_age, '\n  rhat_age :', rhat_age)
# print('\n  slip :', median_slip_corrected, '\n  rhat_slip :', rhat_slip)
print('\n  divergences :', len(mcmc.diagnostics()['divergences']['chain 0']))
print('\n  RMSw on median model : ', rmsw_median_model)


# Plotting 
fig.plot_profile(cl36AMS, param.sig_cl36AMS, height, cl_36_inferred, 'plot_name')
fig.plot_min_max(cl36AMS, height, all_36cl_models, sigAMS=sigAMS, slips=slip_hlines*1e-2)
true_age=true_scenario['ages']
for i in range (0, number_of_events):
    fig.plot_variable_np(all_age[:, i], 'Event '+str(i+1), 'Age', num_fig=i+1)

plt.figure(num=6)
plt.hist(posterior_samples['sigma'], bins=50, color='black')