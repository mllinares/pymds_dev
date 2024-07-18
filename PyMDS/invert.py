#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:18:11 2023

@author: Maureen Llinares
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
import util.post_processing as post
import sys
from seismic_scenario import seismic_scenario as true_scenario
from pyro.infer import MCMC, NUTS
import parameters
import os
import pickle


#%% Initialization
# sys.stdout = open('summary.txt', 'w') # open summary.txt file, all print goes to file
today = datetime.now().strftime("%d/%m/%Y %H:%M:%S") # get today's date day/month/year hour:min:sec

""" Input seismic scenario """
seismic_scenario={} # open new dict to create seismic scenario
number_of_events = 3 # enter the number of event
true_scenario_known = True # true scenario known? (for plotting purpose when using synthetic datafile)
long_term_interval = 500 # interval used in seismic sequence (yr)
seismic_interval = 100 # interval used in seismic sequence (yr)

""" Input parameters"""
param=parameters.param()
cl36AMS = param.cl36AMS
height = param.h
Hfinal = param.Hfinal
trench_depth = param.trench_depth
Hscarp = Hfinal - trench_depth
sigAMS = param.sig_cl36AMS
Data = torch.tensor(cl36AMS)
var_cl36=np.var(cl36AMS)

""" Chose parameters to invert """
invert_slips = True # invert slip array ?
use_rpt = True # use rupture package to find slips
allow_rpt_incertitude = True # Use normal distribution on break-ups
invert_sr = True # invert slip rate ?
invert_quies = False # invert quiescence

if invert_slips==True or allow_rpt_incertitude==True:
    find_slip=True
else:
    find_slip=False
    
""" MCMC parameters, to be set with CAUTION """
tic=time.time()
seed=int(np.random.uniform(low=0, high=1000))
pyro.set_rng_seed(seed) # Random seed
w_step = 10   # number of warmup (~30% of total models)0
nb_sample = 3000 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.7 # target acceptancy probability (<1)

#%% Find slip with rupture package if use_rpt = True
if use_rpt == True:
    slips_rpt = post.precompute_slips(cl36AMS, height, number_of_events-1, plot=True, pen_algo=True, trench_depth=trench_depth)
    seismic_scenario['slips'] = slips_rpt
    number_of_events = len(slips_rpt) # If Pelt algo is used, the number of events can vary
    slip_uncertainty=Hfinal/(number_of_events*2)

#%%Compute geometric scaling factors 
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, number_of_events+1)

#%% Compute long term beforehand in case the SR is known
if invert_sr==False:
    long_term_36cl=forward.long_term(true_scenario, scaling_factors, constants, parameters, long_int = long_term_interval, find_slip = invert_slips)
    
#%% Defining MCMC model
def model(obs):
    
    # Ages: attribute the first age, the other are attributed iteratively to be lower than the previous
    ages = torch.zeros((number_of_events))
    ages[0] = pyro.sample('age1', dist.Uniform(2, 18*1e3))
    for i in range (1, number_of_events):
        max_age = ages[i-1]
        ages[i] = pyro.sample('age'+str(i+1), dist.Uniform(2, max_age-50))  
    seismic_scenario['ages'] = ages
    
    # Slips : can be either infered through MCMC or with precomputation
    if invert_slips == True and use_rpt==False:
        slips = torch.zeros((number_of_events))
        slips[0]=pyro.sample('slip1', dist.Uniform(1, 400))
        for i in range(1, number_of_events):
            max_slip=Hfinal-torch.sum(slips)
            slips[i] = pyro.sample('slip'+str(i+1), dist.Uniform(0, max_slip))
        seismic_scenario['slips'] = slips
        
    # Use rpt package but add some incertitude
    elif allow_rpt_incertitude==True:
        slips = torch.zeros((number_of_events))
        
        for i in range(0, number_of_events):
            slips[i] = pyro.sample('slip'+str(i+1), dist.Uniform(slips_rpt[i]-slip_uncertainty,slips_rpt[i]+slip_uncertainty))
        seismic_scenario['slips'] = slips
        
    # Only rpt package
    elif use_rpt==True and allow_rpt_incertitude==False:
        seismic_scenario['slips'] = slips_rpt
    else:
        seismic_scenario['slips'] = true_scenario['slips']
        
    # SR
    if invert_sr==True:
        seismic_scenario['SR'] = pyro.sample('SR', dist.Uniform(0.1, 4))
    else :
        seismic_scenario['SR'] = true_scenario['SR']
    seismic_scenario['preexp'] = true_scenario['preexp']
    seismic_scenario['quiescence'] = true_scenario['quiescence']
    sigma=pyro.sample('sigma', dist.Uniform(0, var_cl36*1e4)) # incertitude on the model
    
    # The long term is only computed if the SR is searched
    if invert_sr==True:
        t = forward.mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int = long_term_interval, seis_int=seismic_interval, find_slip = find_slip)
    else:
        t = forward.seismic(seismic_scenario, scaling_factors, constants, parameters, Ni=long_term_36cl, seis_int=seismic_interval, find_slip = find_slip)
    return pyro.sample('obs', dist.Normal(t, sigma), obs=obs)

#%% Creating savefile
post.create_save_file(number_of_events, cl36AMS, find_slip, invert_sr)

#%% Running MCMC
tic=time.time()
kernel = NUTS(model, max_tree_depth = tree_depth, target_accept_prob = target_prob) # chose kernel (NUTS, HMC, ...)
mcmc = MCMC(kernel, warmup_steps=w_step, num_samples=nb_sample) 
mcmc.run(obs=Data)
toc=time.time()

# Saving posterior samples
posterior_samples = mcmc.get_samples()
with open('posterior_samples.pickle', 'wb') as handle:
    pickle.dump(posterior_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('diagnosics.pickle', 'wb') as handle:
    pickle.dump(mcmc.diagnostics(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('MCMC done \n start time:', today,'runtime:', '{0:.2f}'.format((toc-tic)/3600), 'hours \n')
print('\n',(toc-tic)/60)

#%% Get MCMC diagnostics and RMSw
rhat_age=post.get_rhat_array('age', mcmc.diagnostics(), number_of_events)
if invert_slips == True or allow_rpt_incertitude==True:
    rhat_slip=post.get_rhat_array('slip', mcmc.diagnostics(), number_of_events)
if invert_sr == True:
    rhat_sr=post.get_rhat('SR', mcmc.diagnostics())
    
#%% Loading sampled results
cl36_profiles = post.load_result(path='results/synthetic_cl36.npy', nb_columns = len(cl36AMS))
ages = post.load_result(path='results/ages.npy', nb_columns = number_of_events)
if find_slip == True :
    slips = post.load_result(path='results/slips.npy', nb_columns = number_of_events)

if invert_sr==True:
    SRs = post.load_result(path='results/SRs.npy')

#%% Get statistics on sampled results
median_age, mean_age, std_age, var_age = post.get_statistics_2D(ages, plot=True, namefig='ages')
median_age70, mean_age70, std_age70, var_age70 = post.get_statistics_2D(ages[int(0.7*len(ages)):len(ages),:], plot=True, namefig='ages70')

if find_slip==True:
    median_slip, mean_slip, std_slip, var_slip = post.get_statistics_2D(slips, plot=True, namefig='slips')
    median_slip70, mean_slip70, std_slip70, var_slip70 = post.get_statistics_2D(slips[int(0.7*len(ages)):len(ages),:], plot=True, namefig='slips70')

if invert_sr==True:
    median_sr, mean_sr, std_sr, var_sr = post.get_statistics_1D(SRs, plot=True, namefig='sr')
    median_sr70, mean_sr70, std_sr70, var_sr70 = post.get_statistics_1D(SRs[int(0.7*len(ages)):len(ages)], plot=True, namefig='sr70')

#%% printing info on the inversion inside summary.txt file
print('\n SUMMARY', param.site_name, '\n ','seed : ', seed, '\n ', 'warmup : ', w_step, '| samples : ', nb_sample, ' \n tree depth :', tree_depth, ' |target acceptancy :', target_prob, '\n\n  age :', median_age70, '\n  rhat_age :', rhat_age)
if find_slip == True:
    print('\n  slip :', median_slip70, '\n  rhat_slip :', rhat_slip)
if invert_sr == True:
    print('\n  SR :', median_sr70, '\n  rhat_SR :', rhat_sr)
if use_rpt == True and allow_rpt_incertitude == False:
    print('\n  slip :', slips_rpt)
print('\n  divergences :', len(mcmc.diagnostics()['divergences']['chain 0']))

#%% Compute median model of the last 30% of infered models
infered_scenario={}
infered_scenario['ages']=torch.tensor(median_age70)
if find_slip==True:
    infered_scenario['slips']=torch.tensor(median_slip70)
elif use_rpt==True and find_slip==False:
    infered_scenario['slips']=slips_rpt
else:
    infered_scenario['slips']=true_scenario['slips']
    
if invert_sr==True:
    infered_scenario['SR']=torch.tensor(median_sr70)
else:
    infered_scenario['SR']=true_scenario['SR']
infered_scenario['preexp']=true_scenario['preexp']
infered_scenario['quiescence']=true_scenario['quiescence']

if invert_sr==False:
    median_cl36_70=forward.seismic(infered_scenario, scaling_factors, constants, parameters, Ni=long_term_36cl, seis_int=seismic_interval, find_slip = find_slip)
else:
    median_cl36_70=forward.mds_torch(infered_scenario, scaling_factors, constants, parameters, long_int = long_term_interval, seis_int=seismic_interval, find_slip = find_slip)

#%% Weighted Root Mean Square Error on models
all_RMSw=np.zeros((len(cl36_profiles)))
for i in range(0, len(cl36_profiles)):
    all_RMSw[i]=post.WRMSE(cl36AMS, cl36_profiles[i], sigAMS)
rmsw_median_model=post.WRMSE(cl36AMS, median_cl36_70.detach().numpy(), incertitude=sigAMS)
print('\n  RMSw on median model : ', rmsw_median_model)
#%% Plotting
# Profiles
post.plot_profile(cl36AMS, param.sig_cl36AMS, height, trench_depth, median_cl36_70, 'plot_median')
post.plot_min_max(cl36AMS, height*1e2, cl36_profiles.T, Hscarp=Hfinal, trench_depth=0, sigAMS=sigAMS, slips=median_slip, plot_name='all_models')
post.plot_min_max(cl36AMS, height*1e2, cl36_profiles[int(0.7*nb_sample)::].T,Hscarp=Hfinal, trench_depth=0, sigAMS=sigAMS, slips=median_slip, plot_name='70_models')

if true_scenario_known==True:
    true_age=true_scenario['ages']
    true_slips=true_scenario['slips']
    true_sr=true_scenario['SR']

#%% Plot ages infered through time
if true_scenario_known == False or number_of_events!=len(true_age):
    for i in range (0, number_of_events):
        post.plot_variable_np(ages[:, i], 'Event '+str(i+1), 'Age', num_fig=i+1) 
else:
    for i in range (0, number_of_events):
        post.plot_variable_np(ages[:, i], 'Event '+str(i+1), 'Age', true_value=true_age[i], num_fig=i+1) #true_value=true_age[i],

#%% Plot slip through time and 2D plots
if invert_slips==True:
    if true_scenario_known == False or number_of_events!=len(true_slips):
        for i in range (0, number_of_events):
            post.plot_variable_np(slips[:, i], title='Slip '+str(i+1), var_name='Slip', num_fig=i+1) 
            post.plot_2D(ages[:,i], slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), median_values=np.array([median_age70[i], median_slip70[i]]))
    elif true_scenario_known == True and number_of_events==len(true_slips):
        for i in range (0, number_of_events):
            post.plot_2D(ages[:,i], slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), true_values=np.array([true_scenario['ages'][i], true_scenario['slips'][i]]), median_values=np.array([median_age70[i], median_slip70[i]]))
            post.plot_variable_np(slips[:, i],true_value=true_slips[i], title='Slip '+str(i+1), var_name='Slip', num_fig=i+1)
    elif true_scenario_known==True and number_of_events!=len(true_age):
        for i in range (0, number_of_events):
            post.plot_2D(ages[:,i], slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1), median_values=np.array([median_age70[i], median_slip70[i]]))
            post.plot_2D(ages[:,i], slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1)+'_with_true_values', true_values=np.array([true_scenario['ages'].numpy(), true_scenario['slips'].numpy()]), median_values=np.array([median_age70[i], median_slip70[i]]))

if invert_sr == True and true_scenario_known==True:
    post.plot_variable_np(SRs, 'SR', 'SR (mm/yr)', true_value = true_scenario['SR'])
elif invert_sr == True and true_scenario_known==False:
    post.plot_variable_np(SRs, 'SR', 'SR (mm/yr)')

#%% Posterior samples
post_ages=post.get_posterior_ages('posterior_samples', number_of_events, nb_sample)
post_ages70=post_ages[int(0.7*len(post_ages))::, :]
median_post_age70, mean_post_age70, std_post_age70, var_post_age70=post.get_statistics_2D(post_ages70, plot=True, namefig='post_ages70')

if find_slip == True:
    post_slips=post.get_posterior_slips('posterior_samples', number_of_events, nb_sample, Hscarp)
    post_slips70=post_slips[int(0.7*len(post_ages))::, :]
    median_post_slip70, mean_post_slip70, std_post_slip70, var_post_slip70=post.get_statistics_2D(post_slips70, plot=True, namefig='post_slips70')
    
if invert_sr==True and true_scenario_known==True:
    post_sr=post.get_posterior_SR('posterior_samples')
    post_sr70=post_sr[int(0.7*len(post_ages))::]
    median_post_sr70, mean_post_sr70, std_post_sr70, var_post_sr70 = post.get_statistics_1D(post_sr70, plot=True, namefig='post_sr70')

#%% Plot slip through time and 2D plots (posterior)
if true_scenario_known == False or number_of_events!=len(true_age):
    for i in range (0, number_of_events):
        post.plot_variable_np(post_ages[:, i], 'Event '+str(i+1)+'_posterior', 'Age', num_fig=i+1) 
else:
    for i in range (0, number_of_events):
        post.plot_variable_np(post_ages[:, i], 'Event '+str(i+1)+'_posterior', 'Age', true_value=true_age[i], num_fig=i+1) #true_value=true_age[i],


if invert_slips==True:
    if true_scenario_known == False or number_of_events!=len(true_slips):
        for i in range (0, number_of_events):
            post.plot_variable_np(post_slips[:, i], title='Slip '+str(i+1)+'_posterior', var_name='Slip', num_fig=i+1) 
            # post.plot_2D(post_ages[:,i], post_slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1)+'_posterior', median_values=np.array([median_post_age70[i], median_post_slip70[i]]))
    elif true_scenario_known == True and number_of_events==len(true_slips):
        for i in range (0, number_of_events):
            # fig.plot_2D(post_ages[:,i], post_slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1)+'_posterior', true_values=np.array([true_scenario['ages'][i], true_scenario['slips'][i]]), median_values=np.array([median_post_age70[i], median_post_slip70[i]]))
            post.plot_variable_np(post_slips[:, i],true_value=true_slips[i], title='Slip '+str(i+1)+'_posterior', var_name='Slip', num_fig=i+1)
    # elif true_scenario_known==True and number_of_events!=len(true_age):
    #     for i in range (0, number_of_events):
    #         fig.plot_2D(post_ages[:,i], post_slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1)+'_posterior', median_values=np.array([median_post_age70[i], median_post_slip70[i]]))
    #         fig.plot_2D(post_ages[:,i], post_slips[:,i], all_RMSw, x_label='age '+str(i+1)+' (yr)',y_label='slip '+str(i+1)+' (cm)', title='age'+str(i+1)+'_vs_slip'+str(i+1)+'_posterior_with_true_values', true_values=np.array([true_scenario['ages'].numpy(), true_scenario['slips'].numpy()]), median_values=np.array([median_post_age70[i], median_post_slip70[i]]))

if invert_sr == True and true_scenario_known==True:
    post.plot_variable_np(post_sr, 'SR_posterior', 'SR (mm/yr)', true_value = true_scenario['SR'])
elif invert_sr == True and true_scenario_known==False:
    post.plot_variable_np(post_sr, 'SR_posterior', 'SR (mm/yr)')
#%% Plots posteriors
# Compute all 36Cl models and save RMSw
cl36_models_post70=np.zeros((len(cl36AMS), len(post_ages70)))
RMSE_post70=np.zeros((len(post_ages70)))
inferred_scenario70={}
inferred_scenario70['preexp'] = true_scenario['preexp']
inferred_scenario70['quiescence'] = true_scenario['quiescence']
for i in range (0, len(post_ages70)):
    inferred_scenario70['ages'] = torch.tensor(post_ages70[i])
    if invert_slips == True:
       inferred_scenario70['slips'] = torch.tensor(post_slips70[i])
    if invert_sr == True:
        inferred_scenario70['SR'] = torch.tensor(post_sr70[i])
    else :
        inferred_scenario70['SR'] = true_scenario['SR']
    cl36_models_post70[:,i]=forward.mds_torch(inferred_scenario70, scaling_factors, constants, parameters, long_int=500, seis_int=200, find_slip = invert_slips)
    RMSE_post70[i]=post.WRMSE(param.cl36AMS, cl36_models_post70[:,i], incertitude=sigAMS)
np.savetxt('cl36_models_post70.txt', cl36_models_post70)
np.savetxt('RMSE_post70.txt', RMSE_post70)
post.plot_min_max(cl36AMS, height*1e2, cl36_models_post70,Hscarp=Hfinal, trench_depth=0, sigAMS=sigAMS, slips=median_slip, plot_name='70_models')
