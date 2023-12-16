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


import matplotlib.pyplot as plt
import numpy as np
import time


name='9m_hmax'
param = parameters.param() # import site parameters
height = param.h
sig = param.sig_cl36AMS

""" First calculate scaling factors """
scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, scaling_factors = geometric_scaling_factors.neutron_scaling(param, constants, 3)


synthetic=forward_function.mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int=500, seis_int=200)

""" Plotting results """
plt.plot(synthetic, height, marker='', linestyle='-', color='orchid', label='Synthetic data')

measured_production = param.cl36AMS
plt.title('Final')

plt.xlabel ('36Cl [at/g]')
plt.ylabel ('Height (m)')
plt.errorbar(param.data[:,64], height, xerr=np.min(param.data[:,64])*0.1, color='black', alpha=0.4,marker='.', linestyle='', label='Synthetic data with noise')
plt.legend()
plt.savefig('synthetic.png', dpi=1200)