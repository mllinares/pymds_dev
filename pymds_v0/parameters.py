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
        self.site_name='9m_numpy_hmax'
        self.rho_rock = 2.66 # rock mean density
        self.rho_coll = 1.5 # colluvium mean density
        self.alpha = 25 # colluvium dip (degrees)
        self.beta = 55 # scarp dip (degrees)
        self.gamma = 35 # eroded scarp dip (degrees)
        self.H_scarp = 1025 # height of preserved scarp (cm)
        self.trench_depth = 0 # trench depth (cm)
        self.long_term_relief = 500 * 1e2 # cumulative height due to long term history (cm)
        self.data = np.loadtxt('data_out.csv', delimiter=',') # samples chemestry
        self.coll = np.loadtxt('coll.txt') # colluvial wedge chemistry
        self.sf = np.loadtxt('sf.txt') # scaling factors for neutrons and muons reactions
        
        
        
        self.cl36AMS = self.data[:, 64]
        self.sig_cl36AMS = self.data[:,65]
        # self.Hfinal = self.H_scarp + self.trench_depth # total height (cm)
        
        self.h  = self.data[:, 62] # position of samples (cm)
        self.Hfinal = max(self.h)
        # self.H_scarp = self.Hfinal
        self.thick = self.data[:, 63]
        self.th2 = (self.thick/2)*self.rho_rock  # 1/2 thickness converted in g.cm-2
        
        # !!! SI ET SEULEMENT SI ON ARRIVE A METTRE LA CONTRAINTE : SUM(SLIPS) = HFINAL
        self.Z = (self.Hfinal - self.h)*self.rho_coll
        
