# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:47:42 2022

@author: Maureen Llinares
"""
import geometric_scaling_factors
import forward_function
import parameters
from constants import constants
from seismic_scenario import seismic_scenario
import add_noise
import torch

import matplotlib.pyplot as plt
import numpy as np
import time


name='9m_hmax'
param = parameters.param() # import site parameters
height = param.h

""" First calculate scaling factors """
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, 10)

""" Calculate the 36Cl """
synthetic=forward_function.mds_new(seismic_scenario, scaling_factors, constants, parameters, 500, 200)
plt.plot(param.data[:,64], height, '.')
plt.plot(synthetic, height)
seismic_scenario['ages']=torch.tensor(seismic_scenario['ages'])
seismic_scenario['slips']=torch.tensor(seismic_scenario['slips'])
synth_torch=forward_function.mds_torch(seismic_scenario, scaling_factors, constants, parameters, 500, 200)
synth_torch2=forward_function.mds_torch_nov(seismic_scenario, scaling_factors, constants, parameters, 500, 200)
plt.plot(synth_torch, height)
plt.plot(synth_torch2, height)
""" Plotting results """
# plt.figure(dpi=1200)
# plt.title('Synthetics '+ name)
# plt.plot(synthetic,  height*1e-2, color = 'cornflowerblue', linestyle='-', marker='', label='smooth synthetic')
# noisy_synth=add_noise.add_noise(synthetic, name, 5*1e4, True, param.data)
# plt.plot(noisy_synth, height*1e-2, marker='.', linestyle='', color='firebrick', label='noisy synthetic')
# plt.xlabel ('36Cl [at/g]')
# plt.ylabel ('Height (m)')
# plt.legend()
# plt.tight_layout()

""" What follows allows you to time the forward funtion """
# nb_iteration=2
# times=np.zeros((nb_iteration))

# plt.figure(num=3)
# plt.plot(synthetic, height*1e-2, marker='o', linestyle='', color='black',  label='cl36 initial')
# plt.figure(num=2)
# sample_nb=np.arange(0, len(height))
# plt.plot(sample_nb, height, color='darkorange', linestyle='', marker='o', label='before execution', )

# for i in range (0, nb_iteration):
    
    # tic=time.time()
    
    # synthetic=forward_function.mds_new(seismic_scenario, scaling_factors, constants, parameters, 500, 200)
    # plt.plot(sample_nb, height, color='darkred', linestyle='', marker='.', label='after execution')
    
    # param = parameters.param()
    # height = param.h
    
    # plt.plot(sample_nb, height, color='black', linestyle='', marker='.', label='reloaded after execution')
    
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # toc=time.time()
    # print('CPU loop ',i, ':', toc-tic, 's')
    # times[i]=toc-tic
    
    # """ Plotting results """
    # plt.figure(num=3)
    # plt.plot(synthetic, height*1e-2, marker='.', linestyle='', color='darkorchid', alpha=0.3, label='cl36 in loop')
    # measured_production = param.cl36AMS
    # plt.title('Final')
    # param = parameters.param() # import site parameters !!!
    # height = param.h
    # plt.plot(synthetic,  height*1e-2, color = 'cornflowerblue', linestyle='', marker='.', label='synthetic')
    # plt.errorbar(measured_production, height*1e-2, xerr=param.sig_cl36AMS, color = 'black', linestyle='', marker='.', label='measured')
    # plt.xlabel ('36Cl [at/g]')
    # plt.ylabel ('Height (m)')
    # plt.tight_layout()