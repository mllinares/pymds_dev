#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:01:12 2023 last modification : 12/16/2023

@author: llinares
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import ruptures as rpt
import os


# %% computational functions
def RMSw(observed_data, modeled_data, incertitude=np.array([])):
    """ Compute the weighted RMS between two datasets
    
    INPUTS : observed_data, 1D array
             modeled_data, 1D array
             incertitude, 1D array
             
    OUTPUT : RMSw, weighted RMS, float"""
    
    if len(incertitude)==0:
        incertitude=np.ones((len(modeled_data))) # to avoid infinity value when incertitude is not known
    rmsw=np.sum(np.sqrt(((observed_data-modeled_data)/incertitude)**2))
    return rmsw
    
def aicc(measurements, calculations, nb_param):
    """ This function allows to calculate the Akaike criterion
    
        INPUTS : measurements, your data, numpy array
                 calculations, your synthetic data, numpy array
                 nb_param, integer
        
        OUTPUT : aicc, Akaike criterion, numpy array
        
        From modelscarp.m, A. Schlagenhauf et al. 2010, adapted for python by M. Llinares
        """

    n = len(measurements) 
    aicc = np.sum((measurements - calculations)**2)
    aicc = n*np.log(aicc/n) + (2*n*nb_param)/(n - nb_param - 1)
    return aicc

def precompute_slips(cl_36, h_samples, nb_bkps, model_name='normal', pen_algo=False, max_bkps=15, plot=False, trench_depth=0):
    """ This function pre-computes the slip array for the inversion (see Truong at al. 2020, rupture package for more info)
    
        INPUTS : cl_36, cl36 concentration data, type : array or tensor, shape : (1, nb_samples)
                 h_samples, height of samples, type :array or tensor, shape : (1, nb_samples)
                 nb_bkps, number of earthquakes, type :integer
                 model_name, name of the model you bwant to use, type : string (l1, l2, normal, rank, rbf, ar), default ='rank'
                 pen_algo, use the penalty algorithm to infer minimum ruptures (experimental), type : boolean, default=False
                 max_bkps, maximum expected number of earthquakes, type : interger, default=5
                 plot, make plots of infered ruptures, type : boolean, default=False
                 
        OUTPUT : slips, slip tensor, torch tensor
                 """
    
    # First we extract the exhumated portion of the scarp
    if trench_depth!=0:
        indexes = np.where(h_samples>trench_depth)[0]
        cl_36_for_rpt = cl_36[indexes]
        h_samples_for_rpt = h_samples[indexes]
        min_height = h_samples[np.min(indexes)-1]
    else:
        cl_36_for_rpt = cl_36
        h_samples_for_rpt = h_samples
        min_height = 0
        
    # Use of dynamic algorithm (no evaluation of minimum number of event)
    if pen_algo==False:
        slips = np.zeros((nb_bkps))
        algo = rpt.Dynp(model=model_name, min_size=10, jump=1).fit(cl_36_for_rpt) # l1, l2, normal, rbf, rank
        result = np.array(algo.predict(n_bkps=nb_bkps))-1
        print(result, len(h_samples))
        h_bkps = np.hstack((h_samples_for_rpt[result][::-1], min_height)) # Get height of break-ups
        print(h_bkps)
        slips = np.zeros((len(result)))  # Define ouput array containing slips

        for i in range (0, len(slips)):
            slips[i]=h_bkps[i]-h_bkps[i+1] # Get slip amount btwn break-ups
        print('slips:', slips, 'sum:', np.sum(slips)) 
        
    if pen_algo == True:
        algo = rpt.Pelt(model=model_name, min_size=10, jump=30).fit(cl_36_for_rpt) # define algo, min_size: min distance btwn change pts, jump : grid of possible change pts
        result = np.array(algo.predict(pen=0.001))-1 # Fit the 36cl signal, pen: penalty value, -1 to get the correct indexes (rpt uses len instead of indexes)
        print(result, len(h_samples))
        h_bkps = np.hstack((h_samples_for_rpt[result][::-1], min_height)) # Get height of break-ups
        print(h_bkps)
        slips = np.zeros((len(result)))  # Define ouput array containing slips

        for i in range (0, len(slips)):
            slips[i]=h_bkps[i]-h_bkps[i+1] # Get slip amount btwn break-ups
        print('slips:', slips, 'sum:', np.sum(slips)) 
    if plot == True:
        plot_rpt(h_samples, cl_36, result, trench_depth=trench_depth)

    slips=torch.tensor(slips.copy()) # reverse slip array, copy.() used to construct torch tensor
    return slips

# %% data management functions
def variable_results(number_of_models, variable, posterior_samples):
    """ Return an array containing the values of posterior samples that are inferred iteratively : e.g. "age"+str(i) = pyro.sample(...)
    Returns also the median and mean values.

    INPUTS: number_of_models, number of models inferred, int
            variable, name of the variable for which you want to save the result, str
            posterior_samples, dictionnary conctaining the posterior samples from pyro MCMC, dict
            
    OUTPUT: save_all, all tested value, 2D array (numpy) shape ((number_of_models, number_of_events))
            median, median value, 1D array (numpy)
            mean, mean value, 1D array (numpy)
            txt file containing tested values
            
    EX : number_of_models= 100, variable ='age', posteterior_samples={[ages1], [ages2], [ages3], ..., [divergences]]}
        =>  save_all  = ([age1_model1, age2_model1, age3_model1],...,[age1_model100, age2_model100, age3_model100])
            median = ([median(age1_model1_100), median(age2_model1_100), median(age3_model1_100)])
            mean = ([mean(age1_model1_100), mean(age2_model1_100), mean(age3_model1_100)]) """
    
    save_all = np.zeros((number_of_models)) # array used to store tested values

    keyword = variable
    inferred = posterior_samples[keyword].detach().numpy()
    save_all = inferred
    median = np.median(inferred)
    mean = np.mean(inferred)
   
    np.savetxt(variable+'.txt', save_all)
    return save_all, median, mean

def array_results(number_of_models, number_of_events, variable, posterior_samples):
    """ Return an array containing the values of posterior samples that are inferred iteratively : e.g. "age"+str(i) = pyro.sample
    Return also the median and mean values.

    INPUTS: number_of_models, number of models inferred, int
            number_of_events, number of events (earthquakes), int
            variable, name of the variable for which you want to save the result, str
            posterior_samples, dictionnary conctaining the posterior samples, dict
            
    OUTPUT: save_all, all tested value, 2D array (numpy) shape ((number_of_models, number_of_events))
            median, median value, 1D array (numpy)
            mean, mean value, 1D array (numpy)
            txt file containing tested values
            
    EX : number_of-models= 100, number_of_events = 3, variable ='age'
        =>  save_all  = ([age1_model1, age2_model1, age3, model1],...,[age1_model100, age2_model100, age3_model100])
            median = ([median(age1_model1-100), median(age2_model1_100), median(age3_model1_100)])
            mean = ([mean(age1_model1-100), mean(age2_model1_100), mean(age3_model1_100)]) """
    
    save_all = np.zeros((number_of_models, number_of_events)) # array used to store tested values
    median = np.zeros((number_of_events))
    mean = np.zeros((number_of_events))

    for i in range (0, number_of_events):
        keyword = variable+str(i+1)
        inferred = posterior_samples[keyword].detach().numpy()
        save_all[:,i] = inferred
        median[i] = np.median(inferred)
        mean[i] = np.mean(inferred)
   
    np.savetxt(variable+'.txt', save_all)
    return save_all, median, mean

def correct_slip_amout(all_slip, mean_slip, median_slip, Hmax):
    """ Correct the slip amount if necessary (when inversing slip amout per event) to avoid ruptures above the sampled portion.
    INPUTS : all_slip, all tested value, 2D array (numpy) shape ((number_of_models, number_of_events))
            mean_slip, mean value, 1D array (numpy)
            median, median value, 1D array (numpy)
            Hmax, maximum height, float
            
   OUTPUTS : all_slip, corrected slips, 2D array (numpy) shape ((number_of_models, number_of_events))
            mean_slip, corrected mean value per event, 1D array (numpy)
            median_slip, corrected median value per event, 1D array (numpy) """
            
    for i in range (0, len(all_slip)):
        if np.sum(all_slip[i])!=Hmax:
            all_slip [i] =(all_slip [i]/np.sum(all_slip [i]))*Hmax
            
    if np.sum(mean_slip)!=Hmax:
        mean_slip = (mean_slip/np.sum(mean_slip))*Hmax
        
    if np.sum(median_slip)!=Hmax:
        median_slip = (median_slip/np.sum(median_slip))*Hmax
    np.savetxt('all_slip_corrected.txt', all_slip)
    return all_slip, mean_slip, median_slip


def get_rhat(variable, diagnostics):
    """ Get the result of the Gilman Rubin filter of a variable (used for display in the nohup.out file)
    INPUTS : variable, variable name used for the MCMC, str
            diagnostics, the diagnostics dictionnary (mcmc.diagnostics), dictionnary
    OUTPUT : rhat, Gilman Rubin filter result (should be close to 1), float"""
    
    rhat=float(diagnostics[variable]['r_hat'].detach().numpy())
    return rhat

def get_rhat_array(variable, diagnostics, nb_event):
    """ Get an array of rhat, useful for ages and slip array (used for display in the summary.txt file)
    INPUTS : variable, variable name used for the MCMC, str
            diagnostics, the diagnostics dictionnary (mcmc.diagnostics), dictionnary
            nb_event, number of events, int
    OUTPUT : array_rhat, array of corresponding rhat values for the infered values, 1D array (numpy)"""
    
    array_rhat=np.zeros((nb_event))
    for i in range (0, nb_event):
        r_hat=get_rhat(variable+str(i+1), diagnostics)
        array_rhat[i]=r_hat  
        
    return array_rhat

def get_70_percent_range(nb_sample, values, save=True, name='no_name'):
    """ Get the last 70% of values (where stability is usually reached)
    INPUTS : nb_sample, the number of samples set for the inversion, type : integer
             values, all of the values, type : numpy 1D array
             save, save the 70 % values or not (a folder is created), default = True, type : bool
             name, name of the saved file, default ='no_name', type : string
    OUTPUTS : the last 70% values, numpy 1D array
              folder named '70_per_cent' and .txt file of values """
        
    values_70 = values[0.3*nb_sample::]
    if save==True:
        if os.path.isdir('70_per_cent')==False:
            os.mkdir('70_per_cent') 
            np.savetxt('70_per_cent/'+name+'.txt', values_70)
        else:
            np.savetxt('70_per_cent/'+name+'.txt', values_70)
    return values_70

def get_statistics(values, plot=False):
    """ Get statistics on values 
    INPUTS : values, all of the values, type : numpy 1D array
    OUTPUTS : median, mean, standard deviation and variancy of the dataset, type : float
              histogram and normal curve of the dataset"""
        
    median=np.median(values)
    mean=np.mean(values)
    std=np.std(values)
    var=np.var(values)
    if plot==True:
        from scipy.stats import norm 
        plt.hist(values, bins=50, color='black')
        plt.plot(values, norm.pdf(values, mean, std)) 
    return median, mean, std, var

# %% plot functions
def plot_profile(clAMS, sigAMS, height, inferred_cl36, plot_name):
    
    """ Plot resulting cl36 profile
    
    INPUTS : clAMS, measured 36Cl (at/g), type : array
             sigAMS, incertitude (if unknown set to 'nan'), type : array
             height, heigth of samples (m), type : array
             inferred_cl36, inferred 36cl, type : array
             plot_name : name of the plot, type : str
             
    OUTPUT : PNG plot """
    
    plt.clf()
    if type(sigAMS)!=str:
        plt.errorbar(clAMS, height*1e-2, xerr=sigAMS, marker='.', linestyle='', color='black', label='Measured concentration')
    else:
        plt.plot(clAMS, height*1e-2, marker='.', linestyle='', color='black', label='Measured concentration')
    plt.plot(inferred_cl36, height*1e-2, color='cornflowerblue', label='Inferred concentration')
    plt.title('Resulting Profile')
    plt.xlabel('[$^{36}$Cl] (at/g)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.savefig(plot_name+'.png', dpi = 1200)
    
def plot_rpt(sample_height, cl36_profile, ruptures, trench_depth=0):
    """ Plot the ruptures found by the rupture package
    INPUTS : sample_height, height of the samples, type : 1D array (torch or numpy)
             cl36_profile, cl36 profile, type : 1D array (torch or numpy)
             ruptures; ruptures found by the ruptures package, float or 1D array """
    plt.clf()
    plt.figure(num=1, figsize=(8.5, 5.3), dpi=1200)
    plt.plot(cl36_profile, sample_height-trench_depth, marker='.', linestyle='', color='black')
    plt.hlines(sample_height[ruptures], min(cl36_profile), max(cl36_profile), linestyle = '--', color='lightsteelblue')
    plt.xlabel('[$^{36}$Cl] (at/g)')
    plt.ylabel('Height (cm)')
    plt.savefig('ruptures.png', dpi=1200)
    
def plot_variable_np(inferred_values, title, var_name, true_value = 1e-38, num_fig = 1):
    
    """ Plot the evolution of a variable through the iteration and histogram of the variable 
    
    INPUTS : inferred_values, all tested values, 1D array
             title, title and name of the plot, str
             var_name, name of the variable, str
             num_fig, figure number, useful only  when plotting mutiple figures, str
             true_value, true value, only useful when known, float 
             
    OUTPUT : plot in PNG format"""
    
    plt.clf()
    # definition of figure
    plt.figure(num=num_fig, dpi=1200, figsize=(8.5, 5.3))
    plt.suptitle(title)
    
    # plot of values over iterations
    plt.subplot(1,2,1)
    plt.title('Values over iterations')
    plt.plot(inferred_values, color='black', label='All values')
    
    # condition to plot the true value
    if true_value!=1e-38:
        plt.hlines(true_value, 0, len(inferred_values), color='seagreen', label='True value')
    # plot median inferred value as horizontal line 
    plt.hlines(np.median(inferred_values), 0, len(inferred_values), color='cornflowerblue', label='Inferred value')
    plt.ylabel(var_name)
    plt.xlabel('Iteration')
    plt.legend()
    
    # histogram of values
    plt.subplot(1,2,2)
    plt.title('Histogram of sampled values')
    bar_heights, bins, patches=plt.hist(inferred_values, bins=50, color='black', label='All values')
    
    # condition to plot the true value
    if true_value!=1e-38:
        plt.vlines(true_value, 0, np.max(bar_heights), color='seagreen', label='True value')
    # plot inferred median value as vertical line on histogram
    plt.vlines(np.median(inferred_values), 0, np.max(bar_heights), color='cornflowerblue', label='Inferred value')
    plt.xlabel(var_name)
    plt.xticks(rotation=45)
    # plt.xlim(0, 800)
    # plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(title+'.png', dpi=1200)
    
    
def plot_min_max(clAMS, height, inferred_cl36, plot_name='min_max', sigAMS=np.array([]), slips=np.array([])):
    
    """ Plot the median model, data, incertitude on data if given, horizontal lines to vizualize the height of the slips and fill
    area of explored models.
    
        INPUT : clAMS, measured 36Cl (at/g), 1D array
                height, height of samples (meters), 1D array
                inferred_cl36, all inferred models, 2D array, shape (number of 36Cl samples, nb of models)
                
                Facultative :
                plot_name, name of your plot, str (default='min_max.png')
                sigAMS, incertitude on 36Cl concentration (at/g), 1D array
                slips, slips associated to each earthquake (meters), 1D array 
                
        OUTPUT : plot saved under the given name, PNG
        """
    plt.clf() # clear all figure
    plt.figure(num=1, dpi=1200, figsize=(8.5, 5.3))
    # define max, min and median model
    max_cl36=np.max(inferred_cl36, axis=1)
    min_cl36=np.min(inferred_cl36, axis=1)
    median_cl36=np.median(inferred_cl36, axis=1)
    
    # condition to plot the errorbar on data or not 
    if len(sigAMS)!=0:
        plt.errorbar(clAMS, height*1e-2, xerr=sigAMS, marker='.', linestyle='', color='black', label='Measured concentration')
    else:
        plt.plot(clAMS, height*1e-2, marker='.', linestyle='', color='black', label='Measured concentration')
    # plot the median model and the greyfill
    plt.plot(median_cl36, height*1e-2, color='cornflowerblue', label='Inferred concentration') # plt the median
    plt.fill_betweenx( height*1e-2, min_cl36, max_cl36, color='cornflowerblue', label='All models', alpha=0.3) # fill between max and min model
    
    # conditions to find the bounds of horizontal lines corresponding to slips
    if min(clAMS)<=min(median_cl36):
        low_bound=min(clAMS)
    else:
        low_bound=min(median_cl36)
        
    if max(clAMS)>=max(median_cl36):
        high_bound=max(clAMS)
    else: 
        high_bound=max(median_cl36)
    
    # condition to decide to plot the horizontal lines
    if len(slips)!=0:
        plt.hlines(slips, low_bound, high_bound, linestyle=':', color='grey')
        
    # labels
    plt.title('Resulting Profile')
    plt.xlabel('[$^{36}$Cl] (at/g)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.savefig(plot_name+'.png', dpi = 1200)

def plot_2D(Var1, Var2, RMSw, x_label, y_label, title, true_values=np.array([]), median_values=np.array([])):
    
    """ Plot of tested values Var1 vs Var2 during inversion, color varying with RMSw
        INPUTS : Var1, variable 1, np.array
                 Var2, variable 2, np.array
                 RMSw, RMS associated to the model from Var1 and Var2, np.array
                 x_label, x label, string
                 y_label, y label, string
                 title, title of the plot, also used in naming the saved figure, string
                 true_values, true values for Var1, Var2, np.array(Var1_true, Var2_true), default=None
                 median_values, inferred values for Var1, Var2, np.array(Var1_true, Var2_true), default=None
        OUTPUT : scatter plot Var1 vs Var2"""
        
    plt.clf() # clear all figure
    plt.figure(figsize=(8.5, 5.3))
    plt.title(title.replace('_', ' '))
    plt.scatter(Var1, Var2, c=RMSw, cmap='viridis_r', alpha=0.5)
    if len(true_values)!=0:
        plt.plot(true_values[0], true_values[1], marker='*', linestyle='', color='firebrick', label='True value', markersize=12)
        plt.legend()
    if len(median_values)!=0:
        plt.plot(median_values[0], median_values[1], marker='*', linestyle='', color='cornflowerblue', label='Infered value', markeredgecolor='black', markeredgewidth=0.5, markersize=12)  
        plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar(label='RMSw')
    plt.savefig(title+'.png', dpi=1200)

# def plot_time_spent(plot_name='CPU'):
#     """ CAUTION : you can use this function only if you used the nohup command to launch the inversion """
    
#     plt.clf()
#     df=pd.read_csv('nohup_out.csv', delimiter=',')
#     df.phase
#     time=df.time
#     df[['hour','min', 'sec']] = df['time'].str.split(':',expand=True)
#     df['hour']=pd.to_numeric(df['hour'])
#     df['min']=pd.to_numeric(df['min'])
#     df['sec']=pd.to_numeric(df['sec'])
#     spent_time=np.array(((df['hour'])+(df['min']/60)+(df['sec']/3600)))
#     plt.plot(spent_time)
#     plt.xlabel('Iteration')
#     plt.ylabel('Time (hour)')
#     lower = np.zeros((100))
#     upper = np.zeros((100))+np.max(spent_time)
#     warm = np.arange(0, 100)
#     plt.fill_between(warm, lower, upper, alpha=0.3, label='Warmup')
#     plt.legend(loc='lower right')
#     plt.savefig(plot_name+'.png', dpi = 1200)
    
def plot_elbow(path_to_files):
    """ Elbow plot to check the minimal number of earthquakes to explain observed data
        INPUT : path_to_file, path to your 36cl modeled data file, str
        OUTPUT : """
