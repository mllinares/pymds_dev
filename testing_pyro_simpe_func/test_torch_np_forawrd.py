#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:56:49 2023

@author: llinares
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pandas as pd
import time
import math

def f_numpy(x, mu, lam):
    # lam2=2*lam
    y=lam*np.exp(-mu/x)
    return y

def f_numpy2(x, mu, lam):
    mu_copy=mu.detach().clone()
    mu_np=mu_copy.detach().numpy()
    lam_copy=lam.detach().clone()
    lam_np=lam_copy.detach().numpy()
    
    y=lam_np*np.exp(-mu_np/x)
    y2=torch.from_numpy(y)
    return y2

def f_torch(xt, mu, lam):
    
    lam2=lam.detach().clone()
    y=lam2*torch.exp(-mu/xt)
    return y

x=np.arange(1, 10, 0.1)
true_l=3
true_m=0.4


x_torch=torch.arange(1, 10, 0.1)

alpha = 0.05 # coefficient de bruit
noise = np.array(np.random.randn(len(x))) * alpha 
data=f_numpy(x, true_m, true_l)+noise
data_torch=f_torch(x_torch, torch.tensor(true_m), torch.tensor(true_l))+noise
plt.plot(data, linestyle='', marker='.', color='black')
df=pd.DataFrame()

def model(obs):
    
    mu= pyro.sample('mu', dist.Uniform(0.01, 10))
    lam= pyro.sample('lam', dist.Uniform(0.1, 40))
    sigma=pyro.sample('sigma', dist.Uniform(0, 5000))
    # sigma = pyro.sample("sigma", dist.Uniform(0,10))
    toto=torch.zeros((3))
    toto[0]=pyro.sample('toto1', dist.Uniform(0.01, 10))
    for i in range(1, len(toto)):
        toto[i]=pyro.sample('toto'+str(i+1), dist.Uniform(0.01, toto[i-1]))
    # TATA=pyro.sample('tata', dist.Uniform(0.01, 20)).expand([3])
    # tata, ind=torch.sort(TATA, descending=True)
    t = f_torch(x_torch, mu, lam)
    return pyro.sample('obs', dist.Normal(t, sigma), obs=data_torch)


def model_2(obs):
    mu= pyro.sample('mu', dist.Uniform(0.01, 10))
    MU=mu.detach().numpy()
    lam= pyro.sample('lam', dist.Uniform(0.1, 40))
    LAM =lam.detach().numpy()
    t = f_numpy(x, MU, LAM)
    T=torch.tensor(t)
    return pyro.sample('obs', dist.Normal(T, 0.5), obs=data_torch)

def model_3(obs):
    mu= pyro.sample('mu', dist.Uniform(0.01, 10))
    lam= pyro.sample('lam', dist.Uniform(0.1, 10))
    sigma=pyro.sample('sigma', dist.Uniform(0, 5000))
    
    
    # lam2=lam.clone().detach()
    # sigma = pyro.sample("sigma", dist.Uniform(0,10))
    # toto=torch.zeros((3))
    # toto[0]=pyro.sample('toto1', dist.Uniform(0.01, 10))
    # for i in range(1, len(toto)):
    #     toto[i]=pyro.sample('toto'+str(i+1), dist.Uniform(0.01, toto[i-1]))
    # TATA=pyro.sample('tata', dist.Uniform(0.01, 20)).expand([3])
    # tata, ind=torch.sort(TATA, descending=True)
    t = f_numpy2(x, mu, lam)
    # return pyro.sample('obs', dist.Normal(t, 2), obs=data_torch)
    return pyro.sample('obs', dist.Normal(t, sigma), obs=data_torch)
    

""" usage MCMC """
tic=time.time()
w_step=30
n_sample=100
tic=time.time()
pyro.set_rng_seed(50)
kernel = NUTS(model, jit_compile=True) # , jit_compile=True
mcmc = MCMC(kernel, warmup_steps=w_step, num_samples=n_sample)
mcmc.run(obs=data_torch)
toc=time.time()
print('CPU:', toc-tic)
# mcmc.print_summary()
posterior_sample = mcmc.get_samples()
toc=time.time()
# print(mcmc.diagnostics())
# print(posterior_sample)

""" Plotting """
synth_m=torch.mean(posterior_sample['mu'][w_step::])
synth_l=torch.mean(posterior_sample['lam'][w_step::])
ynth_m=torch.median(posterior_sample['mu'])
synth_l=torch.median(posterior_sample['lam'])
synth=f_numpy(x, synth_m.detach().numpy(), synth_l.detach().numpy())
# plt.plot(synth, linestyle='', marker='o', color='orchid', alpha=0.3)
plt.figure(num=1)
plt.plot(synth)
# s=3*15*(np.exp(2*15/np.arange(1, 10, 0.1)))

plt.figure(num=2, figsize=(10, 5))
plt.suptitle('MU, true='+str(true_m))
plt.subplot(1,2,1)
plt.hlines(true_m, 0, len(posterior_sample['mu']), color='seagreen', label='True value')
plt.hlines(synth_m, 0, len(posterior_sample['mu']), color='firebrick', label='Inferred value')
plt.plot(posterior_sample['mu'], color='black')
lower=np.zeros((w_step))+min(posterior_sample['mu'].detach().numpy())-0.1
upper=np.zeros((w_step))+max(posterior_sample['mu'].detach().numpy())
warm=np.arange(0, w_step)
plt.fill_between(warm, lower, upper, alpha=0.3)
plt.ylabel('Mu Value')
plt.xlabel('Iteration')
plt.legend(loc='upper right')
plt.subplot(1,2,2)
plt.vlines(true_m, 0, len(posterior_sample['mu']), color='seagreen', label='True value')
plt.vlines(synth_m, 0, len(posterior_sample['mu']), color='firebrick', label='Inferred value')
plt.hist(posterior_sample['mu'], bins=10, color='silver', label='all models')
plt.hist(posterior_sample['mu'][0:w_step], bins=10, alpha=0.3, label='warm up models')
plt.hist(posterior_sample['mu'][w_step::], bins=10, color='black', label='Samples')
plt.ylabel('Density of models')
plt.xlabel('Mu Value')
plt.legend(loc='upper right')

plt.figure(num=3, figsize=(10, 5))
plt.suptitle('LAMBDA, true='+str(true_l))
plt.subplot(1,2,1)
plt.hlines(true_l, 0, len(posterior_sample['lam']), color='seagreen', label='True value')
plt.hlines(synth_l, 0, len(posterior_sample['lam']), color='firebrick', label='Inferred value')
plt.plot(posterior_sample['lam'], color='black')
lower=np.zeros((w_step))+min(posterior_sample['lam'].detach().numpy())-0.1
upper=np.zeros((w_step))+max(posterior_sample['lam'].detach().numpy())
warm=np.arange(0, w_step)
plt.fill_between(warm, lower, upper, alpha=0.3)
plt.ylabel('Lambda Value')
plt.xlabel('Iteration')
plt.legend(loc='upper right')
plt.subplot(1,2,2)
plt.hist(posterior_sample['lam'], bins=10, color='silver', label='all models')
plt.hist(posterior_sample['lam'][0:w_step], bins=10, alpha=0.3, label='warm up models')
plt.hist(posterior_sample['lam'][w_step::], bins=10, color='black', label='Samples')
plt.vlines(true_l, 0, len(posterior_sample['lam']), color='seagreen', label='True value')
plt.vlines(synth_l, 0, len(posterior_sample['lam']), color='firebrick', label='Inferred value')
plt.ylabel('Density of models')
plt.xlabel('Lambda Value')
plt.legend(loc='upper right')

print(math.dist(data, synth))
""" RMS for each model """
RMS=np.zeros((w_step+n_sample))
synthetic_models=np.zeros((n_sample, len(synth)))
for i in range(0, n_sample):
    synthetic_models[i, :]=f_numpy(x, posterior_sample['mu'][i], posterior_sample['lam'][i])

# plt.plot(synthetic_models)