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
seismic_scenario['ages'] = torch.tensor([7000, 2500, 500]) # exhumation ages, older to younger (yr)
# seismic_scenario['ages'] = torch.tensor([6443.1968, 5138.0879, 4698.4023,  153.3770])
seismic_scenario['slips'] = torch.tensor([300, 300, 300]) # slip corresponding to the events (cm)
seismic_scenario['SR'] = 0.3 # long term slip rate of your fault (mm/yr)
seismic_scenario['preexp'] = 5*1e3 # Pre-expositionn period (yr)
seismic_scenario['start_depth'] = seismic_scenario['preexp'] * seismic_scenario['SR'] * 1e-1 # (cm) along the fault plane
seismic_scenario['erosion_rate'] = param.erosion_rate # Erosion rate (mm/yr)
seismic_scenario['quiescence'] = 0*1e3 # Quiescence period (yr), must be older than last event


""" The quiscence period and trench depth will modify the ages and slip arrays : 
    
    The quiescence is defined as a period of low or no activity before the sesimic sequence 
       => We add the quiescence to the ages array corresponding to lack of slip
    The trench depth corresponds to a slip with no  age:
        => we add a event with age = 0 and a slip corresponding to the trench depth
        
    """
# Handling of quiescence period
if seismic_scenario['quiescence'] !=0 :
    seismic_scenario['ages'] = np.hstack((seismic_scenario['quiescence'] + seismic_scenario['ages'][0], seismic_scenario['ages']))
    seismic_scenario['slips'] = np.hstack((0, seismic_scenario['slips']))


# Handling of trench height
if param.trench_depth !=0 :
    seismic_scenario['ages'] = np.hstack((seismic_scenario['ages'], 0))
    seismic_scenario['slips'] = np.hstack((seismic_scenario['slips'], param.trench_depth))

""" Errors and Warnings handling """

# Viability of seismic history
# if len(seismic_scenario['ages'])!=len(seismic_scenario['slips']):
#     print(Fore.RED+'Error:')
#     print(Fore.CYAN+'File : seismic_scenario.py')
#     print(Style.RESET_ALL+' number of events don\'t match number of slips \n Exiting')
#     sys.exit()
    
# if np.sum(seismic_scenario['slips'])>param.Hfinal :
#     print(Fore.RED+'Error:')
#     print(Fore.CYAN+'File : seismic_scenario.py')
#     print(Style.RESET_ALL+'Cumulative slips is higher than the scarp \n Exiting')
#     sys.exit()
    

# # Warning : tells the user that the ages array was not ordered, the program continues but orders the array     
# is_sorted = lambda a: np.all(a[:len(a)] <= a[1:]) # function to check if an array is sorted

# if is_sorted(seismic_scenario['ages'][::len(seismic_scenario['ages'])])==False: 
#     seismic_scenario['ages'][::len(seismic_scenario['ages'])].sort() # Sort ages array
#     print(Fore.MAGENTA+'WARNING:')
#     print(Fore.CYAN+'File : seismic_scenario.py')
#     print(Style.RESET_ALL+'Ages array was not sorted, slips might not correspond to event ages, \nSorting ages array \nRunning...' )
#     print(Style.RESET_ALL+'Abort crtl+c')

