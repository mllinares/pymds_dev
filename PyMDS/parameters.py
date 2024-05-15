#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:39:21 2023

@author: llinares
"""

import numpy as np


class param:
    """ This class contains all the site parameters needed for the inversion """
    def __init__(self):
        self.site_name='ma3'
        self.rho_rock = 2.66 # rock mean density
        self.rho_coll = 1.5 # colluvium mean density
        self.alpha = 30 # colluvium dip (degrees)
        self.beta = 45 # scarp dip (degrees)
        self.gamma = 30 # eroded scarp dip (degrees)
        self.erosion_rate = 0 # Erosion rate (mm/yr)
        self.trench_depth = 457 # trench depth (cm)
        self.long_term_relief = 500 * 1e2 # cumulative height due to long term history (cm)
        self.data = np.loadtxt('data.csv', delimiter=',') # samples chemestry
        self.coll = np.loadtxt('coll.csv', delimiter=',') # colluvial wedge chemistry
        self.sf = np.loadtxt('sf.csv', delimiter=',') # scaling factors for neutrons and muons reactions
        
        self.cl36AMS = self.data[:, 64]
        self.sig_cl36AMS = self.data[:,65]
        self.h  = self.data[:, 62] # position of samples (cm)
        self.Hfinal = max(self.h)
        # self.Hfinal = self.Hfinal + self.trench_depth # total height (cm)
        self.thick = self.data[:, 63]
        self.th2 = (self.thick/2)*self.rho_rock  # 1/2 thickness converted in g.cm-2
        self.Z = (self.Hfinal - self.h)*self.rho_coll
        
