#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:18:11 2023

@author: llinares
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


""" Input seismic scenario """
seismic_scenario={}
erosion_rate = 0 # Erosion rate (mm/yr)
number_of_events = 3
seismic_scenario['erosion_rate'] = erosion_rate

""" Input parameters"""
param=parameters.param()
cl36AMS = param.cl36AMS
height = param.h
Data = torch.tensor(cl36AMS)


""" Geometric scaling """
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, number_of_events+1)

""" MCMC parameters """
pyro.set_rng_seed(50)
w_step = 100 # number of warmup (~30% of total models)
nb_sample = 300 # number of samples

""" MCMC model """
def model(obs):
    
    ages = torch.zeros((number_of_events))
    ages[0] = pyro.sample('age1', dist.Uniform(2.0, 30*1e3))

    for i in range (1, number_of_events):
        max_age = ages[i-1]
        ages[i] = pyro.sample('age'+str(i+1), dist.Uniform(2.0, max_age))  
    # print('\n age', ages)
    seismic_scenario['ages'] = ages
    seismic_scenario['SR'] = true_scenario['SR']
    seismic_scenario['preexp'] = true_scenario['preexp']
    slips = torch.tensor(true_scenario['slips'])
    seismic_scenario['quiescence'] = true_scenario['quiescence']
    seismic_scenario['slips'] = slips
    sigma=pyro.sample('sigma', dist.Uniform(0, 10000)) # helps for the inference
    
    t = forward.mds_torch(seismic_scenario, scaling_factors, constants, parameters, 500, 200)
    return pyro.sample('obs', dist.Normal(t, sigma), obs=obs)

""" usage MCMC """
kernel = NUTS(model, max_tree_depth = 6, target_accept_prob = 0.7) # chose kernel (NUTS, HMC, ...)
mcmc = MCMC(kernel, warmup_steps=w_step, num_samples=nb_sample) 
mcmc.run(obs=Data)
# print(mcmc.diagnostics())

""" plotting and saving """
posterior_samples = mcmc.get_samples()

# Includes the warmup steps
a1_model = posterior_samples['age1']
a2_model = posterior_samples['age2']
a3_model = posterior_samples['age3']

# Save all models 
save_all = np.zeros((len(a1_model), 3))
save_all[:, 0] = a1_model
save_all[:, 1] = a2_model
save_all[:, 2] = a3_model
np.savetxt('all_ages.txt', save_all)

# Models without the warmup
a1 = posterior_samples['age1'][w_step::]
a2 = posterior_samples['age2'][w_step::]
a3 = posterior_samples['age3'][w_step::]

# Save models without warmup
save_age=np.zeros((len(a1), 3))
save_age[:, 0] = a1
save_age[:, 1] = a2
save_age[:, 2] = a3
np.savetxt('age_no_w.txt', save_age)

# Plotting variables throught post_process.py
true_age = true_scenario['ages']
for i in range (0, number_of_events):
    fig.plot_variable('age'+str(i+1), true_age[i], np.median(save_all[:,i]), posterior_samples, w_step, nb_sample, i+1)

# Plotting inferred scenario
inferred_scenario={}
inferred_scenario['ages']=np.array([torch.median(a1_model), torch.median(a2_model), torch.median(a3_model)])
inferred_scenario['SR'] = true_scenario['SR']
inferred_scenario['preexp'] = true_scenario['preexp']
inferred_scenario['slips'] = true_scenario['slips']
inferred_scenario['quiescence'] = true_scenario['quiescence']
inferred_scenario['erosion_rate'] = true_scenario['erosion_rate']

cl_36_inferred=forward.mds_new(inferred_scenario, scaling_factors, constants, parameters, 500, 200)
fig.plot_profile(cl36AMS, 'nan', height, cl_36_inferred, 'plot_name', 4)

