# -*- coding: utf-8 -*-
"""

@author: Maureen Llinares, adapted for python from:
    
    Schlagenhauf A., Gaudemer Y., Benedetti L., Manighetti I., Palumbo L.,
    Schimmelpfennig I., Finkel R., Pou K.
    G.J.Int., 2010
    
    Tesson & Benedetti, Quat. Geol., 2019
    
"""

import numpy as np
from math import pi, sin
from chemistry_scaling import clrock, clcoll
import time

def mds_torch(seismic_scenario, scaling_factors, constants, parameters, long_int=100, seis_int=100, find_slip=False):
    
    """ Process the long term history of the fault 
        INPUTS  : seismic_scenario, seismic scenario (see seismic_scenario.py), dtype: dictionary
                  scaling_factors, calculated with geometric_scaling_factors.py, dtype: dictionary
                  constants, global constants (see constants.py), dtype: dictionary
                  parameters, site parameters (see parameters.py), dtype : class parameter
                  long_int, time interval for the calculation of 36Cl due to longterm exposure, dtype : int
                  seis_int,  time interval for the calculation of 36Cl due to longterm exposure, dtype : int
                  find_slip, set to 'True' ONLY if you want to find the amount of slip per earthquake, dtype : bool
                     
       OUTPUT : cl36_long_term, inhertited cl36 concentration, dtype: numpy array"""
       
    import torch
    
    """ VARIABLES INITIALIZATION """
    # Site parameters
    param = parameters.param()
    time_interval = long_int
    Hfinal = param.Hfinal
    alpha = param.alpha
    beta = param.beta
    EL = param.sf
    data = param.data
    coll = param.coll
    thick = param.thick
    th2 = param.th2  # 1/2 thickness converted in g.cm-2
    h = param.h.copy()  # initial positions of the samples at surface (cm)
    Z = param.Z.copy()
    rho_rock = param.rho_rock
    rho_coll = param.rho_coll
    epsilon = param.erosion_rate
    Hscarp = Hfinal-param.trench_depth
    
    # Seismic scenario
    age_base = seismic_scenario['ages']
    age = age_base.clone().detach().numpy() 
    # Save age
    load_ages=np.load('results/ages.npy')
    save_ages=np.concatenate((load_ages, age))
    np.save('results/ages.npy', save_ages)
    
    slip_base = seismic_scenario['slips']
    slip = slip_base.clone().detach().numpy() 
    slip = np.abs(slip)
    # cumsum(slip) must be equal to Hfinal to avoid estimation of slip in non sampled part
    if (find_slip == True and np.sum(slip)<Hscarp) or (find_slip == True and np.sum(slip))>Hscarp:
        slip = ((slip/np.sum(slip))*Hscarp)+1 # +1 because int(slips) is used below and sometimes the last sample is not included
        # Saving slips
    load_slips=np.load('results/slips.npy')
    save_slips=np.concatenate((load_slips, slip))
    np.save('results/slips.npy', save_slips)
    # Handling of quiescence period
    if seismic_scenario['quiescence'] !=0 :
        age = np.hstack((seismic_scenario['quiescence'] + age[0], age))
        slip = np.hstack((0, slip))

    # Handling of trench height
    if param.trench_depth !=0 :
        age = np.hstack((age, 0))
        slip = np.hstack((slip, param.trench_depth))
     
    SR = seismic_scenario['SR']
    # Saving SR
    load_SRs=np.load('results/SRs.npy')
    save_SRs=np.hstack((load_SRs, SR.clone().detach().numpy()))
    np.save('results/SRs.npy', save_SRs)
    preexp = seismic_scenario['preexp']
    
    # attenuation constants from gscale
    Lambda_f_e = scaling_factors['Lambda_f_e']
    so_f_e = scaling_factors['so_f_e']
    Lambda_f_diseg = scaling_factors['Lambda_f_diseg']
    so_f_diseg = scaling_factors['so_f_diseg']
    Lambda_f_beta_inf = scaling_factors['Lambda_f_beta_inf']
    so_f_beta_inf = scaling_factors['so_f_beta_inf']
    
    # Constant
    lambda36 = constants['lambda36']
    
    # Loading of Earth magnetic field variations from file 'EL'
    EL_f = EL[:,2]  # scaling factor for neutrons (S_el,f)
    EL_mu = EL[:,3]  # scaling factor for muons (S_el,mu)
    
    # Other useful variables
    N_eq = len(age)  # number of earthquakes
    R = np.sum(slip)  # total cumulative slip
    Rc = np.cumsum(slip)   
    Rc = np.hstack((0, Rc)) # slip added up after each earthquake
    d = data  # substitution of matrix data by matrix d
    d[:,62] = Z  # samples position along z
    d[:,63] = thick * rho_rock  # thickness converted in g.cm-2
    slip_gcm2 = slip*rho_coll  # coseismic slip in g.cm-2
    sc = np.cumsum(slip_gcm2)  # cumulative slip after each earthquake (g.cm-2)
    sc0 = np.hstack((0, sc)) 

    # Positions along e initially (eo)
    eo = np.zeros(len(Z)) 

    for iseg in range (0, N_eq):
        eo[np.where((Z > sc0[iseg]) & (Z <= sc0[iseg + 1]))] = epsilon*age[iseg]*0.1*rho_rock # in g.cm-2
    
    eo[0:len(Z)]=epsilon*age[0]*0.1*rho_rock
    eo = eo + th2  # we add the 1/2 thickness : sample position along e is given at the sample center
    
    if preexp==0:
        N_in = np.zeros(len(Z))  
        Ni = np.zeros(len(Z))  
        Nf = np.zeros(len(Z)) 
     
    """-----------------------------PRE-EXPOSURE PROFILE-------------------------
     Modified version of Pre-exposure calculation including a erosion rate of the upper surface (TESSON 2015)

     Calculation of [36Cl] concentration profile at the end of pre-exposure.

     initialization at 0"""
     
    No = np.zeros(len(Z))  # No : initial concentration (here = zero) before pre-exposure
    Ni = np.zeros(len(Z))    # Ni :  
    Nf = np.zeros(len(Z))  # Nf : final 36Cl concentration 

    N_out = 0
    N_in = 0
    P_rad = 0
    P_cosmo = 0
    tic1=time.time()
    # conversion denud rate from m/Myr to cm/yr
    SR = SR*1e-1 # (cm/yr)
    start_depth = preexp * SR # (cm) along the fault plane
    xa=np.zeros((len(data),2))
    for j in range (0, len(data)) : # loop over samples

        dpj = d[j,:]
        d0 = dpj.copy() # .detach().clone() qui a changé la forme du profil
        d0[62] = 0 
        dpj[62] = ((dpj[62].copy())*sin((beta - alpha)*pi/180))+(start_depth*sin((beta - alpha)*pi/180)*rho_coll)  # in the direction perpendicular to colluvium surface
        N_in = No[j]  # initial concentration (here = zero)
     
        # B2 - LOOP - iteration on time (ii) during pre-exposure
        for ii in range (0, int(preexp/long_int)): 
            
            P_cosmo, P_rad = clrock(d[j,:],eo[j], Lambda_f_e, so_f_e, EL_f[1], EL_mu[1]) 
            # scaling at depth due to the presence of the colluvium: scoll=Pcoll(j)/Pcoll(z=0)
            P_coll = clcoll(coll, dpj, Lambda_f_diseg[0], so_f_diseg[0] ,EL_f[1], EL_mu[1]) 
            P_zero = clcoll(coll, d0, Lambda_f_beta_inf ,so_f_beta_inf, EL_f[1],EL_mu[1]) 
            scoll = P_coll/P_zero  
            P_tot = P_rad + P_cosmo*scoll # only P (Pcosmogenic) is scalled by scoll
            N_out = N_in + (P_tot - lambda36*N_in)*time_interval  # minus radioactive decrease during same time step
            N_in = N_out
            dpj[62] = dpj[62] - (SR*time_interval*sin((beta - alpha)*pi/180)*rho_coll) # depth update

        Ni[j] = N_out
        xa[j, 0]=Ni[j]
        xa[j, 1]=dpj[62].copy()
    toc1=time.time()    
    """ VARIABLE INITIALIZATION """
    time_interval=seis_int
    # site parameters (see site_parameters.py)
    param = parameters.param()
    alpha = param.alpha # colluvial wedge slope
    beta = param.beta # fault-plane slope
    Hfinal = param.Hfinal # total post-glacial height of the fault-plane, must include the depth of sample taken below the collucial wedge surface.
    rho_coll = param.rho_coll # colluvial wedge mean density
    rho_rock = param.rho_rock  # rock sample mean density
    thick = param.thick
    th2 = param.th2 # 1/2 thickness converted in g.cm-2
    h = param.h.copy()  # initial positions of the samples at surface (cm)- integer
    Z = param.Z.copy()
    data = param.data
    coll = param.coll
    EL = param.sf
    
    # Scaling factors
    S_S = scaling_factors['S_S']
    Lambda_f_e = scaling_factors['Lambda_f_e']
    so_f_e = scaling_factors['so_f_e']
    Lambda_f_diseg = scaling_factors['Lambda_f_diseg']
    so_f_diseg = scaling_factors['so_f_diseg']
    Lambda_f_beta_inf = scaling_factors['Lambda_f_beta_inf']
    so_f_beta_inf = scaling_factors['so_f_beta_inf']
    EL_f = EL[:,2]  # scaling factor for neutrons (S_el,f)
    EL_mu = EL[:,3]  # scaling factor for muons (S_el,mu)
   
    # Other useful variables
    N_eq = len(age)  # number of earthquakes
    R = np.sum(slip)  # total cumulative slip
    Rc = np.cumsum(slip)   
    Rc = np.hstack((0, Rc)) # slip added up after each earthquake
    d = data.copy() # substitution of matrix data by matrix d
    d[:,62] = Z.copy()  # samples position along z
    d[:,63] = ((thick)*rho_rock).copy()  # thickness converted in g.cm-2
    slip_gcm2 = slip * rho_coll  # coseismic slip in g.cm-2
    sc = np.cumsum(slip_gcm2)  # cumulative slip after each earthquake (g.cm-2)
    sc0 = np.hstack((0, sc)) 
    Nf = np.zeros(len(Z))  # Nf : final 36Cl concentration
    
    """ SEISMIC PHASE
    
    the term 'segment' is used for the samples exhumed by an earthquake-
    Calculation of [36Cl] profiles during seismic cycle.
    
    Separated in two stages : 
      1) When samples are at depth and progressively rising because of earthquakes
         (moving in the direction z with their position in direction e fixed)
      2) When samples are brought to surface and only sustaining erosion
         (moving along the direction e)
         
    FIRST EXHUMED SEGMENT is treated alone."""
    
    eo = np.zeros(len(Z)) 

    for iseg in range (0, N_eq):
       eo[np.where((Z > sc0[iseg]) & (Z <= sc0[iseg + 1]))] = epsilon*age[iseg]*0.1*rho_rock # in g.cm-2
    eo[-1] = epsilon*age[0]*0.1*rho_rock
    eo = eo + th2  # we add the 1/2 thickness : sample position along e is given at the sample center
    j1 = np.where((Z >= sc0[0]) & (Z <= sc0[1]))[0]
    
    # print(j1)# Averf samples from first exhumed segment 
    N1 = np.zeros(len(Z[j1])) 
    tic2=time.time()
    # C1 - Loop - iteration on samples (k) from first exhumed segment
    for k in range (0, len(j1)):
        
        djk = d[j1[k],:]
        hjk = h[j1[k]].copy()   # position of sample k (cm)
        N_in = float(Ni[j1[k]])  # initial concentration is Ni, obtained after pre-exposure
        ejk = eo[j1[k]]   # initial position along e is eo(j1(k)) 
        # print(h[j1[k]])
        # C2 - Loop - iteration on  time steps ii from t1 (= age eq1) to present
        for ii in range (0, int(age[0]), time_interval):
            
            P_cosmo, P_rad = clrock(djk, ejk, Lambda_f_e, so_f_e, EL_f[1], EL_mu[1]) 
            scorr = S_S[int(hjk)-1]/S_S[0]     
            
            P_tot = P_rad + P_cosmo*scorr# only Pcosmogenic is scaled with scorr
            N_out = N_in + (P_tot - lambda36*N_in)*time_interval  # minus radioactive decrease during same time step
            
            ejk = ejk - epsilon*time_interval*0.1*rho_rock  # new position along e at each time step (g.cm-2)
            N_in = N_out  
        N1[k] = N_out # AVERF

    Nf[j1] = N1 

    # ITERATION ON SEGMENTS 2 to N_eq
    # print(h)
    # h=param.h.copy()
    # C3 - Loop - iteration on each segment (from segment 2 to N_eq=number of eq)
    for iseg in range (1, N_eq):
        
        j = np.where((Z > sc0[iseg]) & (Z <= sc0[iseg+1]))[0]  # index of samples from segment iseg
        z_j = Z[j]  # initial depth along z of these samples (g.cm-2)
        N_new = np.zeros(len(z_j))
        
        # C4 - Loop - iteration each sample from segment iseg     
        for k in range (0, len(j)) :                                                  
            
            ejk = eo[j[k]].copy()  # initial position along e is stil eo.
            djk = d[j[k],:].copy()
            djk[62] = djk[62]*sin((beta - alpha)*pi/180) # AVERF 
            
            N_in = Ni[j[k]]  #  initial concentration is Ni
            
            # C5 - Loop - iteration on previous earthquakes
            for l in range (0, iseg):                                                     
             
                # depth (along z) are modified after each earthquake
                djk[62] = djk[62] - (slip[l]*rho_coll*sin((beta - alpha) *pi/180))# AVERF 
                d0 = djk.copy()
                d0[62] = 0 

                #------------------------------            
                # C6 - DEPTH LOOP - iteration during BURIED PERIOD (T1 -> T(iseg-1))
                #------------------------------ 
                
                
                for iii in range (0, int(age[l]-age[l+1]), time_interval):
                    P_cosmo,P_rad = clrock(djk, ejk, Lambda_f_e, so_f_e, EL_f[1], EL_mu[1]) 
                    # scaling at depth due to the presence of the colluvium: scoll=Pcoll(j)/Pcoll(z=0)
                    P_coll = clcoll(coll, djk, Lambda_f_diseg[l+1], so_f_diseg[l+1], EL_f[1], EL_mu[1]) 
                    P_zero = clcoll(coll, d0, Lambda_f_beta_inf, so_f_beta_inf, EL_f[1], EL_mu[1]) 
                    scoll = P_coll/P_zero  
                    
                    P_tot = P_rad + P_cosmo*scoll # only P (Pcosmogenic) is scalled by scoll
                    N_out = N_in + (P_tot - lambda36*N_in)*time_interval # minus radioactive decrease during same time step
                    N_in = N_out
                N_in = N_out  
            N_in = N_out 
            hjk = h[j[k]].copy()
          
            #------------------------------         
            # C7 - SURFACE LOOP - iteration during EXHUMED PERIOD 
            # h=param.h.copy()
            
            for ii in range (0, int(age[iseg]), time_interval):
                # print(hjk)
                P_cosmo,P_rad = clrock(djk,ejk,Lambda_f_e,so_f_e,EL_f[1],EL_mu[1]) 
                scorr = S_S[int(hjk)-1]/S_S[0]  # surface scaling factor (scorr)
                P_tot = P_rad + P_cosmo*scorr  # only Pcosmogenic is scaled with scorr
                N_out = N_in + (P_tot - lambda36*N_in)*time_interval # minus radioactive decrease during same time step
                ejk = ejk - epsilon*time_interval*0.1*rho_rock  # new position along e at each time step (g.cm-2)
                N_in = N_out
            N_new[k] = N_out
        Nf[j] = N_new 
    Nf2=torch.tensor(Nf)
    
    # Writting final concentration file
    cl_load=np.load('results/synthetic_cl36.npy')
    cl_save=np.concatenate((cl_load, Nf))
    np.save('results/synthetic_cl36.npy', cl_save)
    return Nf2

def long_term(seismic_scenario, scaling_factors, constants, parameters, long_int, find_slip):
    import torch
    
    """ VARIABLES INITIALIZATION """
    # Site parameters
    param = parameters.param()
    time_interval = long_int
    Hfinal = param.Hfinal
    # facet_height = param.facet_height
    alpha = param.alpha
    beta = param.beta
    EL = param.sf
    data = param.data
    coll = param.coll
    thick = param.thick
    th2 = param.th2  # 1/2 thickness converted in g.cm-2
    h = param.h.copy()  # initial positions of the samples at surface (cm)
    Z = param.Z.copy()
    rho_rock = param.rho_rock
    rho_coll = param.rho_coll
    epsilon = param.erosion_rate
    Hscarp = Hfinal-param.trench_depth
    
    # Seismic scenario
    # age_base = seismic_scenario['ages']
    # age = age_base.clone().detach().numpy() 
    # slip_base=seismic_scenario['slips']
    # slip = slip_base.clone().detach().numpy() 
    
    # # cumsum(slip) must be equal to Hfinal to avoid estimation of slip in non sampled part
    # if (find_slip == True and np.sum(slip)<Hscarp) or (find_slip == True and np.sum(slip))>Hscarp:
    #     slip = ((slip/np.sum(slip))*Hscarp)+1 # +1 because int(slips) is used below and sometimes the last sample is not included
    
    # # Handling of quiescence period
    # if seismic_scenario['quiescence'] !=0 :
    #     age = np.hstack((seismic_scenario['quiescence'] + age[0], age))
    #     slip = np.hstack((0, slip))

    # Handling of trench height
    # if param.trench_depth !=0 :
    #     age = np.hstack((age, 0))
    #     slip = np.hstack((slip, param.trench_depth))
     
    SR = seismic_scenario['SR']
    preexp = seismic_scenario['preexp']
    
    # attenuation constants from gscale
    Lambda_f_e = scaling_factors['Lambda_f_e']
    so_f_e = scaling_factors['so_f_e']
    Lambda_f_diseg = scaling_factors['Lambda_f_diseg']
    so_f_diseg = scaling_factors['so_f_diseg']
    Lambda_f_beta_inf = scaling_factors['Lambda_f_beta_inf']
    so_f_beta_inf = scaling_factors['so_f_beta_inf']
    
    # Constant
    lambda36 = constants['lambda36']
    
    # Loading of Earth magnetic field variations from file 'EL'
    EL_f = EL[:,2]  # scaling factor for neutrons (S_el,f)
    EL_mu = EL[:,3]  # scaling factor for muons (S_el,mu)
    
    # Other useful variables
    # N_eq = len(age)  # number of earthquakes
    # R = np.sum(slip)  # total cumulative slip
    # Rc = np.cumsum(slip)   
    # Rc = np.hstack((0, Rc)) # slip added up after each earthquake
    d = data  # substitution of matrix data by matrix d
    d[:,62] = Z  # samples position along z
    d[:,63] = thick * rho_rock  # thickness converted in g.cm-2
    # slip_gcm2 = slip*rho_coll  # coseismic slip in g.cm-2
    # sc = np.cumsum(slip_gcm2)  # cumulative slip after each earthquake (g.cm-2)
    # sc0 = np.hstack((0, sc)) 

    # Positions along e initially (eo)
    eo = np.zeros(len(Z)) 

    # for iseg in range (0, N_eq):
    #     eo[np.where((Z > sc0[iseg]) & (Z <= sc0[iseg + 1]))] = epsilon*age[iseg]*0.1*rho_rock # in g.cm-2
    
    # eo[-1]=epsilon*age[0]*0.1*rho_rock
    # eo = eo + th2  # we add the 1/2 thickness : sample position along e is given at the sample center
    
    if preexp==0:
        N_in = np.zeros(len(Z))  
        Ni = np.zeros(len(Z))  
        Nf = np.zeros(len(Z)) 
     
    """-----------------------------PRE-EXPOSURE PROFILE-------------------------
     Modified version of Pre-exposure calculation including a erosion rate of the upper surface (TESSON 2015)

     Calculation of [36Cl] concentration profile at the end of pre-exposure.

     initialization at 0"""
     
    No = np.zeros(len(Z))  # No : initial concentration (here = zero) before pre-exposure
    Ni = np.zeros(len(Z))    # Ni :  
    Nf = np.zeros(len(Z))  # Nf : final 36Cl concentration 

    N_out = 0
    N_in = 0
    P_rad = 0
    P_cosmo = 0

    # conversion denud rate from m/Myr to cm/yr
    SR = SR*1e-1 # (cm/yr)
    start_depth = preexp*SR #+Hfinal #facet_height    #(cm) along the fault plane
    # print(start_depth/1e2, 'm')
    # preexp = (facet_height-Hfinal) / SR
    # print(preexp/1e3, 'Kyr')
    
    xa=np.zeros((len(data),2))
    # print(int((preexp-age[0])/long_int))
    for j in range (0, len(data)) : # loop over samples

        dpj = d[j,:]
        d0 = dpj.copy() # .detach().clone() qui a changé la forme du profil
        d0[62] = 0 
        dpj[62] = ((dpj[62].copy())*sin((beta - alpha)*pi/180))+(start_depth*sin((beta - alpha)*pi/180)*rho_coll)  # in the direction perpendicular to colluvium surface
        N_in = No[j]  # initial concentration (here = zero)
     
        # B2 - LOOP - iteration on time (ii) during pre-exposure
        for ii in range (0, int(preexp), long_int): 
            
            P_cosmo, P_rad = clrock(d[j,:],eo[j], Lambda_f_e, so_f_e, EL_f[1], EL_mu[1]) 
            # scaling at depth due to the presence of the colluvium: scoll=Pcoll(j)/Pcoll(z=0)
            P_coll = clcoll(coll, dpj, Lambda_f_diseg[0], so_f_diseg[0] ,EL_f[1], EL_mu[1]) 
            P_zero = clcoll(coll, d0, Lambda_f_beta_inf ,so_f_beta_inf, EL_f[1],EL_mu[1]) 
            scoll = P_coll/P_zero  
            P_tot = P_rad + P_cosmo*scoll # only P (Pcosmogenic) is scalled by scoll
            N_out = N_in + (P_tot - lambda36*N_in)*time_interval  # minus radioactive decrease during same time step
            N_in = N_out
            dpj[62] = dpj[62].copy() - (SR*time_interval*sin((beta - alpha)*pi/180)*rho_coll) # depth update

        Ni[j] = N_out
        xa[j, 0]=Ni[j]
        xa[j, 1]=dpj[62].copy()
    lg=xa[:,0]
    return lg

def seismic(seismic_scenario, scaling_factors, constants, parameters, Ni, seis_int, find_slip):
    import torch
    time_interval=seis_int
    # site parameters (see site_parameters.py)
    param = parameters.param()
    alpha = param.alpha # colluvial wedge slope
    beta = param.beta # fault-plane slope
    Hfinal = param.Hfinal # total post-glacial height of the fault-plane, must include the depth of sample taken below the collucial wedge surface.
    rho_coll = param.rho_coll # colluvial wedge mean density
    rho_rock = param.rho_rock  # rock sample mean density
    thick = param.thick
    th2 = param.th2 # 1/2 thickness converted in g.cm-2
    h = param.h.copy()  # initial positions of the samples at surface (cm)- integer
    Z = param.Z.copy()
    data = param.data
    coll = param.coll
    EL = param.sf
    Hscarp = Hfinal-param.trench_depth
    epsilon = param.erosion_rate
    
    # Scaling factors
    S_S = scaling_factors['S_S']
    Lambda_f_e = scaling_factors['Lambda_f_e']
    so_f_e = scaling_factors['so_f_e']
    Lambda_f_diseg = scaling_factors['Lambda_f_diseg']
    so_f_diseg = scaling_factors['so_f_diseg']
    Lambda_f_beta_inf = scaling_factors['Lambda_f_beta_inf']
    so_f_beta_inf = scaling_factors['so_f_beta_inf']
    EL_f = EL[:,2]  # scaling factor for neutrons (S_el,f)
    EL_mu = EL[:,3]  # scaling factor for muons (S_el,mu)
    
    # Seismic scenario
    age_base = seismic_scenario['ages']
    age = age_base.clone().detach().numpy()
    # Save age
    load_ages=np.load('results/ages.npy')
    save_ages=np.concatenate((load_ages, age))
    np.save('results/ages.npy', save_ages)
    slip_base=seismic_scenario['slips']
    slip = slip_base.clone().detach().numpy() 
    slip = np.abs(slip)
    # Constant
    lambda36 = constants['lambda36']
    
    # cumsum(slip) must be equal to Hfinal to avoid estimation of slip in non sampled part
    if (find_slip == True and np.sum(slip)<Hscarp) or (find_slip == True and np.sum(slip))>Hscarp:
        slip = ((slip/np.sum(slip))*Hscarp)+1 # +1 because int(slips) is used below and sometimes the last sample is not included
        # Saving slips
    load_slips=np.load('results/slips.npy')
    save_slips=np.concatenate((load_slips, slip))
    np.save('results/slips.npy', save_slips)
    # Handling of quiescence period
    if seismic_scenario['quiescence'] !=0 :
        age = np.hstack((seismic_scenario['quiescence'] + age[0], age))
        slip = np.hstack((0, slip))

    # Handling of trench height
    if param.trench_depth !=0 :
        age = np.hstack((age, 0))
        slip = np.hstack((slip, param.trench_depth))
   
    # Other useful variables
    N_eq = len(age)  # number of earthquakes
    R = np.sum(slip)  # total cumulative slip
    Rc = np.cumsum(slip)   
    Rc = np.hstack((0, Rc)) # slip added up after each earthquake
    d = data.copy() # substitution of matrix data by matrix d
    d[:,62] = Z.copy()  # samples position along z
    d[:,63] = (thick)*rho_rock  # thickness converted in g.cm-2
    slip_gcm2 = slip * rho_coll  # coseismic slip in g.cm-2
    sc = np.cumsum(slip_gcm2)  # cumulative slip after each earthquake (g.cm-2)
    sc0 = np.hstack((0, sc)) 
    Nf = np.zeros(len(Z))  # Nf : final 36Cl concentration
    
    
    
    """ SEISMIC PHASE
    
    the term 'segment' is used for the samples exhumed by an earthquake-
    Calculation of [36Cl] profiles during seismic cycle.
    
    Separated in two stages : 
      1) When samples are at depth and progressively rising because of earthquakes
         (moving in the direction z with their position in direction e fixed)
      2) When samples are brought to surface and only sustaining erosion
         (moving along the direction e)
         
    FIRST EXHUMED SEGMENT is treated alone."""
    
    eo = np.zeros(len(Z)) 

    for iseg in range (0, N_eq):
       eo[np.where((Z > sc0[iseg]) & (Z <= sc0[iseg + 1]))] = epsilon*age[iseg]*0.1*rho_rock # in g.cm-2
    eo[0:len(Z)] = epsilon*age[0]*0.1*rho_rock
    eo = eo + th2  # we add the 1/2 thickness : sample position along e is given at the sample center
    j1 = np.where((Z >= sc0[0]) & (Z <= sc0[1]))[0]  # Averf samples from first exhumed segment 
    N1 = np.zeros(len(Z[j1])) 
    
    # C1 - Loop - iteration on samples (k) from first exhumed segment
    for k in range (0, len(j1)):
        
        djk = d[j1[k],:].copy()
        hjk = h[j1[k]].copy()   # position of sample k (cm)
        N_in = float(Ni[j1[k]])  # initial concentration is Ni, obtained after pre-exposure
        ejk = eo[j1[k]]   # initial position along e is eo(j1(k)) 
             
        # C2 - Loop - iteration on  time steps ii from t1 (= age eq1) to present
        for ii in range (0, int(age[0]),time_interval):
            
            P_cosmo, P_rad = clrock(djk, ejk, Lambda_f_e, so_f_e, EL_f[1], EL_mu[1]) 
            scorr = S_S[int(hjk-1)]/S_S[0]     
            
            P_tot = P_rad + P_cosmo*scorr            # only Pcosmogenic is scaled with scorr
            N_out = N_in + (P_tot - lambda36*N_in)*time_interval  # minus radioactive decrease during same time step
            
            ejk = ejk - epsilon*time_interval*0.1*rho_rock  # new position along e at each time step (g.cm-2)
            N_in = N_out  
        N1[k] = N_out # AVERF

    Nf[j1] = N1 

    # ITERATION ON SEGMENTS 2 to N_eq
    
    # C3 - Loop - iteration on each segment (from segment 2 to N_eq=number of eq)
    for iseg in range (1, N_eq):
            
        j = np.where((Z > sc0[iseg]) & (Z <= sc0[iseg+1]))[0]  # index of samples from segment iseg
        z_j = Z[j]  # initial depth along z of these samples (g.cm-2)
        N_new = np.zeros(len(z_j))
        
        # C4 - Loop - iteration each sample from segment iseg     
        for k in range (0, len(j)) :                                                  
            
            ejk = eo[j[k]]  # initial position along e is stil eo.
            djk = d[j[k],:]
            djk[62] = djk[62]*sin((beta - alpha)*pi/180) # AVERF 
            
            N_in = Ni[j[k]]  #  initial concentration is Ni
            
            # C5 - Loop - iteration on previous earthquakes
            for l in range (0, iseg):                                                     
             
                # depth (along z) are modified after each earthquake
                djk[62] = djk[62].copy() - (slip[l]*rho_coll*sin((beta - alpha) *pi/180)) # AVERF 
                d0 = djk.copy()
                d0[62] = 0 

                #------------------------------            
                # C6 - DEPTH LOOP - iteration during BURIED PERIOD (T1 -> T(iseg-1))
                #------------------------------ 
                for iii in range (0, int((age[l]-age[l+1])/time_interval)):
                    P_cosmo,P_rad = clrock(djk, ejk, Lambda_f_e, so_f_e, EL_f[1], EL_mu[1]) 
                    # scaling at depth due to the presence of the colluvium: scoll=Pcoll(j)/Pcoll(z=0)
                    P_coll = clcoll(coll, djk, Lambda_f_diseg[l+1], so_f_diseg[l+1], EL_f[1], EL_mu[1]) 
                    P_zero = clcoll(coll, d0, Lambda_f_beta_inf, so_f_beta_inf, EL_f[1], EL_mu[1]) 
                    scoll = P_coll/P_zero  
                    
                    P_tot = P_rad + P_cosmo*scoll # only P (Pcosmogenic) is scalled by scoll
                    N_out = N_in + (P_tot - lambda36*N_in)*time_interval # minus radioactive decrease during same time step
                    N_in = N_out
                N_in = N_out  
            N_in = N_out 
            hjk = h[j[k]]
          
            #------------------------------         
            # C7 - SURFACE LOOP - iteration during EXHUMED PERIOD 
            
            for ii in range (0, int(age[iseg]), time_interval):
                P_cosmo,P_rad = clrock(djk,ejk,Lambda_f_e,so_f_e,EL_f[1],EL_mu[1]) 
                scorr = S_S[1+int(hjk)]/S_S[0]  # surface scaling factor (scorr)
                P_tot = P_rad + P_cosmo*scorr  # only Pcosmogenic is scaled with scorr
                N_out = N_in + (P_tot - lambda36*N_in)*time_interval # minus radioactive decrease during same time step
                ejk = ejk - epsilon*time_interval*0.1*rho_rock  # new position along e at each time step (g.cm-2)
                N_in = N_out
            N_new[k] = N_out
        Nf[j] = N_new 
    Nf2=torch.tensor(Nf)
    
    cl_load=np.load('results/synthetic_cl36.npy')
    # print(len(cl_save), len(Nf))
    cl_save=np.concatenate((cl_load, Nf))
    np.save('results/synthetic_cl36.npy', cl_save)
    
    
    return Nf2