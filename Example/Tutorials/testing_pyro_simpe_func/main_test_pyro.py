#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:25:24 2023

@author: llinares
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import util.post_process as post

""" Input seismic scenario """
a=20
b=6000

x=np.arange(-100, 100)
x_torch=torch.tensor(x)

def function(x_val, slope, y_inter):
    y_val=(slope*x_val)+y_inter
    y_val_torch=torch.tensor(y_val)
    return y_val_torch
""" Input parameters"""
y=function(x, a, b)

"""Adding some noise"""
alpha = 200 # noise coeff
np.random.seed(0) # to reproduce the same noise eaxh time!
noise = np.random.randn(len(x)) * alpha # adding random noise


Data = y+noise
variancy=np.var(y.detach().numpy())
print(variancy)
""" MCMC parameters """
pyro.set_rng_seed(300)
w_step = 10 # number of warmup (~30% of total models)
nb_sample = 1000 # number of samples

""" MCMC model """
def model(obs):
    
    infer_a=pyro.sample('a', dist.Uniform(2.0, 100))
    infer_b=pyro.sample('b', dist.Uniform(2.0, 100*1e3))
    sigma=pyro.sample('sigma', dist.Uniform(0, variancy))
    t = function(x_torch, infer_a, infer_b)
    return pyro.sample('obs', dist.Normal(t, sigma), obs=obs)

""" usage MCMC """
kernel = NUTS(model, max_tree_depth = 10, target_accept_prob = 0.9) # chose kernel (NUTS, HMC, ...)
mcmc = MCMC(kernel, warmup_steps=w_step, num_samples=nb_sample) 
mcmc.run(obs=Data)
print('MCMC done \n')

""" Post processing """
posterior_samples = mcmc.get_samples()

""" plotting and saving with post_process.py """
# Saving
all_slope, median_slope, mean_slope = post.value_results('a', posterior_samples)
all_y_inter, median_y_inter, mean_y_inter = post.value_results('b', posterior_samples)
post.plot_variable_np(all_slope, 'title', 'slope', fig_num=1, clear=False)
post.plot_variable_np(all_y_inter, 'title', 'y_inter', fig_num=2, clear=False)

# Compute all models
all_models=np.zeros((len(x), len(all_slope)))
for i in range(0, len(all_slope)):
    all_models[:,i]=function(x, all_slope[i], all_y_inter[i])
# post.plot_min_max(x, y, all_models, fig_num=3, clear=False)

# Compute and plot the median model 
inferred_yval=function(x, median_slope, median_y_inter)
post.plot_profile(x, y+noise, inferred_yval, fig_num=3, clear=False)


# Get r_hat value
rhat_a=post.get_rhat('a', mcmc.diagnostics())
rhat_b=post.get_rhat('b', mcmc.diagnostics())

# printing info 
print('\n SUMMARY',  ' \n ', 'warmup : ', w_step, ' samples : ', nb_sample, '\n\n  a :', median_slope, '\n  rhat_a :', rhat_a, '\n\n  b :', median_y_inter, '\n  rhat_a :', rhat_b)
print('\n  divergences :', len(mcmc.diagnostics()['divergences']['chain 0']))


# # Plotting 
# post.plot_variable_np(all_slope, 'slope', 'slope', num_fig=1, true_value=a)
# post.plot_variable_np(all_y_inter, 'y_inter', 'y_inter', num_fig=2, true_value=b)

