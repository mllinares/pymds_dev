# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:45:08 2022

@author: Maureen Llinares

Set the seismic scenario for the forward function
"""
# import numpy as np
import numpy as np
import torch
import parameters 
import sys
from colorama import Fore, Back, Style # allows to color text for errors


# import the cumulative height to calculate the preexp or SR
param=parameters.param()
cumulative_height = param.long_term_relief

# Seismic scenario stored in a dict
seismic_scenario={}
# seismic_scenario['ages'] = np.array([ 6987.4707,  4441.9854,  1256.0417, 500]) # exhumation ages, older to younger (yr)
seismic_scenario['ages'] = torch.tensor([9000, 4000, 1500]) # exhumation ages, older to younger (yr)
# seismic_scenario['ages'] = torch.tensor([6443.1968, 5138.0879, 4698.4023,  153.3770])
seismic_scenario['slips'] = torch.tensor([300, 300, 290]) # slip corresponding to the events (cm)
seismic_scenario['SR'] = 0.8 # long term slip rate of your fault (mm/yr)
seismic_scenario['preexp'] = 50*1e3 # Pre-expositionn period (yr)
seismic_scenario['start_depth'] = seismic_scenario['preexp'] * seismic_scenario['SR'] * 1e-1 # (cm) along the fault plane
seismic_scenario['erosion_rate'] = param.erosion_rate # Erosion rate (mm/yr)
seismic_scenario['quiescence'] = 0*1e3 # Quiescence period (yr), must be older than last event


