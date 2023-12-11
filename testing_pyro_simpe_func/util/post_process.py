#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:01:12 2023

@author: llinares
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# import pandas as pd
import numpy as np
import torch

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
        INPUTS : measuremments, your data, numpy array
                 calculations, your synthetic data, numpy array
                 nb_param, integer
        
        OUTPUT : aicc, Akaike criterion, numpy array
        
        From modelscarp.m, A. Schlagenhauf et al. 2010, adapted for python by M. Llinares
        """

    n = len(measurements) 
    aicc = np.sum((measurements - calculations)**2)
    aicc = n*np.log(aicc/n) + (2*n*nb_param)/(n - nb_param - 1)
    return aicc

# %% data management functions
def value_results (value_name, posterior_samples):
    all_value=posterior_samples[value_name].detach().numpy()
    median_value=np.median(all_value)
    mean_value=np.mean(all_value)
    return all_value, median_value, mean_value

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

def get_rhat(variable, diagnostics):
    """ Get the result of the Gilman Rubin filter of a variable (used for display in the nohup.out file)
    INPUT : variable, variable name used for the MCMC, str
            diagnostics, the diagnostics dictionnary (mcmc.diagnostics), dictionnary
    OUTPUT : rhat, Gilman Rubin filter result (should be close to 1), float"""
    
    rhat=float(diagnostics[variable]['r_hat'].detach().numpy())
    return rhat

def get_rhat_array(variable, diagnostics, nb_event):
    """ Get an array of rhat, useful for ages and slip array (used for display in the nohup.out file)
    INPUT : variable, variable name used for the MCMC, str
            diagnostics, the diagnostics dictionnary (mcmc.diagnostics), dictionnary
            nb_event, number of events, int
    OUTPUT : array_rhat, array of corresponding rhat values for the infered values, 1D array (numpy)"""
    
    array_rhat=np.zeros((nb_event))
    for i in range (0, nb_event):
        r_hat=get_rhat(variable+str(i+1), diagnostics)
        array_rhat[i]=r_hat  
        
    return array_rhat

# %% plot functions
def plot_profile(x_values, data, inferred, plot_name='plot_profile', sigma=np.array([]), fig_num=1, clear=True):
    
    """ Plot resulting cl36 profile
    
    INPUTS : clAMS, measured 36Cl (at/g), type : array
             sigAMS, incertitude (if unknown set to 'nan'), type : array
             height, heigth of samples (m), type : array
             inferred_cl36, inferred 36cl, type : array
             plot_name : name of the plot, type : str
             
    OUTPUTS : PNG plot """
    
    if clear==True:
        plt.clf()
        
    plt.figure(num=fig_num)
    if len(sigma)!=0:
        plt.errorbar(x_values, data, xerr=sigma, marker='.', linestyle='', color='black', label='Measured concentration')
    else:
        plt.plot(x_values, data, marker='.', linestyle='', color='black', label='Measured concentration')
    plt.plot(x_values, inferred, color='cornflowerblue', label='Inferred concentration')
    plt.title('Resulting Plot')
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.savefig(plot_name+'.png', dpi = 1200)
    
    
def plot_variable_np(inferred_values, title, var_name, true_value = 1e-38, fig_num = 1, clear=True):
    
    """ Plot the evolution of a variable through the iteration and histogram of the variable 
    
    INPUTS : inferred_values, all tested values, 1D array
             title, title and name of the plot, str
             var_name, name of the variable, str
             num_fig, figure number, useful only  when plotting mutiple figures, str
             true_value, true value, only useful when known, float 
             
    OUTPUT : plot in PNG format"""
    
    if clear ==True: 
        plt.clf()
    # definition of figure
    plt.figure(num=fig_num, dpi=1200, figsize=(8.5, 5.3))
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
    plt.savefig(title+'.png')
    
    
def plot_min_max(x_values, data, inferred, plot_name='min_max', sigma=np.array([]), fig_num=1, clear=True):
    
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
    if clear==True:
        plt.clf() # clear all figure
    plt.figure(num=fig_num, dpi=1200, figsize=(8.5, 5.3))
    # define max, min and median model
    max_cl36=np.max(inferred, axis=1)
    min_cl36=np.min(inferred, axis=1)
    median_cl36=np.median(inferred, axis=1)
    
    # condition to plot the errorbar on data or not 
    if len(sigma)!=0:
        plt.errorbar(x_values, data, xerr=sigma, marker='.', linestyle='', color='black', label='Measured concentration')
    else:
        plt.plot(x_values, data, marker='.', linestyle='', color='black', label='Measured concentration')
    # plot the median model and the greyfill
    plt.plot(x_values, median_cl36, color='cornflowerblue', label='Inferred concentration') # plt the median
    plt.fill_betweenx( x_values, min_cl36, max_cl36, color='cornflowerblue', label='All models', alpha=0.3) # fill between max and min model
    
    # labels
    plt.title('Resulting Profile min max')
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.savefig(plot_name+'.png', dpi = 1200)

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