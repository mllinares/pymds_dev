# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:25:39 2024

@author: maure
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import pickle
import itertools
import ruptures as rpt
import torch

#%%
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
        algo = rpt.Dynp(model=model_name, min_size=2, jump=30).fit(cl_36_for_rpt) # l1, l2, normal, rbf, rank
        result = np.array(algo.predict(n_bkps=nb_bkps))-1
        print(result, len(h_samples))
        h_bkps = np.hstack((h_samples_for_rpt[result][::-1], min_height)) # Get height of break-ups
        print(h_bkps)
        slips = np.zeros((len(result)))  # Define ouput array containing slips

        for i in range (0, len(slips)):
            slips[i]=h_bkps[i]-h_bkps[i+1] # Get slip amount btwn break-ups
        print('slips:', slips, 'sum:', np.sum(slips)) 
        
    if pen_algo == True:
        algo = rpt.Pelt(model=model_name, min_size=2, jump=30).fit(cl_36_for_rpt) # define algo, min_size: min distance btwn change pts, jump : grid of possible change pts
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

#%% Result file management (sampled models)
def create_save_file(number_of_events, cl36AMS, invert_slips=False, invert_SR=False):
    """ Initialize .npy result files that are filled during the inversion for inferred 36Cl, ages, (slips and SR if inverted).
    
        INPUTS : number_of_events, number of events, type (int)
                 cl36AMS, measured 36Cl, type (2D np.array)
                 invert_slips, set to True if slips are inverted, default: False, type (bool)
                 invert_SR, set to True if slips are inverted, default: False, type (bool)"""
    
    # Check if results files already exists, print warning and overwrite if True, create result folder if False
    if os.path.isdir('results')==False:
        os.mkdir('results')
    elif os.path.isdir('results')==True:
        print("Result files found: Overwritting preexisting result files!!")
        
    # Create the files (filled during MCMC)
    np.save('results/synthetic_cl36.npy', cl36AMS) # First line is the data
    np.save('results/ages.npy', np.ones((number_of_events))) 
    if invert_slips==True:
        np.save('results/slips.npy', np.ones((number_of_events)))
    if invert_SR==True:
        np.save('results/SRs.npy', np.ones((number_of_events)))
    return
    
def load_result(path, nb_columns=0):
    """ Load .npy result file saved during MCMC and the first line (used only to initialize the files).
    Warning : contains warmup AND sampling steps !!!
    Returns np.array of the variable.
        
        INPUTS : path, path to file, type (str)
                 nb_columns, number of columns used to reshape the array, only required for multi-dim arrays, type (int) 
        OUTPUT : inferred_variable, variable, type (np.array, 1D or 2D)"""
        
    inferred_variable = np.load(path) # Load files
    
    # Reshaping the arrays, if multi-dim and suppressing first line (used to create the file in the first place)
    if nb_columns!=0:
        inferred_variable = inferred_variable.reshape(int(len(inferred_variable)/nb_columns), nb_columns) # reshape
        inferred_variable = inferred_variable[1::, :]
    else:
        inferred_variable = inferred_variable[1::]
    return inferred_variable

def npy2csv(file_name):
    """Convert a .npy to csv and save it under 'csv_results' folder
    INPUT : filename without extension, type (str) """
    
    array_to_convert = np.load(file_name+'.npy')
    if os.path.isdir('csv_results')==False:
        os.mkdir('csv_results')
    np.savetxt('csv_results'+file_name+'.csv', array_to_convert, delimiter=',')
    return

#%% Result file management (posterior models)
def get_posterior_ages(dict_name, nb_events, nb_samples):
    """ Get posterior values of ages (from mcmc.posterior_samples dict)
        INPUT : dict_name, name of the pickle file containing 'posterior_samples' dictionnary, type (str)
                nb_events, number of events, type (int)
                nb_samples, number of samples set in MCMC, type (int)
        OUTPUT : ages, posterior ages, type (2D np.array)"""
    
    # Open the dictionnary 
    with open(dict_name+'.pickle', 'rb') as handle:
        posterior = pickle.load(handle)
    
    # Get the values in a np.array
    ages=np.zeros((nb_samples, nb_events))
    for i in range (0, nb_events):
        ages[:,i]=posterior['age'+str(i+1)].detach().numpy()
    return ages

def get_posterior_slips(dict_name, nb_events, nb_samples, Hscarp):
    """ Get posterior values of slips (from mcmc.posterior_samples dict)                   
        INPUT : dict_name, name of the pickle file containing 'posterior_samples' dictionnary, type (str)
                nb_events, number of events, type (int)
                nb_samples, number of samples set in MCMC, type (int)
                Hscarp, height of the scarp (without trench depth), type (int)
                
        OUTPUT : slips, posterior ages, type (2D np.array)"""
    # Open the dictionnary 
    with open(dict_name+'.pickle', 'rb') as handle:
        posterior = pickle.load(handle)
        
    slips=np.zeros((nb_samples, nb_events))
    for i in range (0, nb_events):
        slips[:,i]=posterior['slip'+str(i+1)].detach().numpy()
    # Slips are corrected so they do not exceed the maximum height of the scarp
    for j in range (0, len(slips)):
        slips[j,:] = ((slips[j,:]/np.sum(slips[j,:]))*Hscarp)+1
    return  slips

def get_posterior_SR(dict_name):
    """ Get posterior values of SR (from mcmc.posterior_samples dict)
        INPUT : dict_name, name of the pickle file containing 'posterior_samples' dictionnary, type (str)
                nb_events, number of events, type (int)
                nb_samples, number of samples set in MCMC, type (int)
        OUTPUT : SR, posterior slip rates, type (1D np.array)"""
    with open(dict_name+'.pickle', 'rb') as handle:
        posterior = pickle.load(handle)
    SR=posterior['SR'].detach().numpy()
    return SR
    
#%% Diagnostic tools
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

def get_statistics_1D(values, plot=False, namefig='sts'):
    """ Get statistics on values 
    INPUTS : values, all of the values, type : numpy 1D array
    OUTPUTS : median, mean, standard deviation and variancy of the dataset, type : float
              histogram and normal curve of the dataset"""
    
    median=np.median(values)
    mean=np.mean(values)
    std=np.std(values)
    var=np.var(values)
    if plot==True:
        plt.clf()
        # from scipy.stats import norm 
        bar_heights, bins, patches=plt.hist(values, bins=50, color='black', alpha=0.3)
        plt.vlines(median, 0, np.max(bar_heights), color='cornflowerblue', label='median : '+"{:.2f}".format(median))
        plt.vlines(mean, 0, np.max(bar_heights), color='crimson', label='mean : '+"{:.2f}".format(mean))
        plt.vlines(median+std,0, np.max(bar_heights), color='white', alpha=0, label='std : '+"{:.2f}".format(std))
        plt.legend(loc='upper right')
        # plt.plot(values, norm.pdf(values, mean, std)) 
        if os.path.isdir('plots_stats')==False:
            os.mkdir('plots_stats')

        plt.savefig('plots_stats/'+namefig+'.png', dpi=1200)
    return median, mean, std, var

def get_statistics_2D(array, plot=False, namefig='sts'):
    
    """Get statistics on columns in 2D array (i.e. ages and slips)
    INPUTS : array,  2D np.array
             plot, decide to plots figure, type (bool)
             numfig, name of the plot, type (str), default='sts'
    OUTPUTS : median, mean, std, var as 1D array """
    
    #Initialize output arrays : ex median=[median_1st_ev, median_2nd_ev,..., median_last_ev]
    median=np.zeros((np.shape(array)[1]))
    mean=np.zeros((np.shape(array)[1]))
    std=np.zeros((np.shape(array)[1]))
    var=np.zeros((np.shape(array)[1]))
    
    # Fill the arrays
    for i in range(0, np.shape(array)[1]):
        median[i], mean[i], std[i], var[i]=get_statistics_1D(array[:,i], plot=plot, namefig=namefig+str(i))
    return median, mean, std, var

def get_possibles(stacked_array):
    
    arrays=stacked_array.tolist()
    all_combinations=list(itertools.product(*arrays))
    return all_combinations
    
#%% Plotting functions 
def plot_profile(clAMS, sigAMS, height, trench_depth, inferred_cl36, plot_name):
    
    """ Plot resulting cl36 profile
    
    INPUTS : clAMS, measured 36Cl (at/g), type : array
             sigAMS, incertitude (if unknown set to 'nan'), type : array
             height, heigth of samples (m), type : array
             inferred_cl36, inferred 36cl, type : array
             plot_name : name of the plot, type : str
             
    OUTPUT : PNG plot """
    
    plt.clf()
    if type(sigAMS)!=str:
        plt.errorbar(clAMS, (height-trench_depth)*1e-2, xerr=sigAMS, marker='.', linestyle='', color='black', label='Measured concentration')
    else:
        plt.plot(clAMS, (height-trench_depth)*1e-2, marker='.', linestyle='', color='black', label='Measured concentration')
    plt.plot(inferred_cl36, (height-trench_depth)*1e-2, color='cornflowerblue', label='Inferred concentration')
    plt.title('Resulting Profile')
    plt.xlabel('[$^{36}$Cl] (at/g)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.savefig(plot_name+'.png', dpi = 1200)
    plt.close('all')
    return
    
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
    plt.close('all')
    return
    
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
    # plt.tight_layout()
    plt.savefig(title+'.png', dpi=1200)
    plt.close('all')
    return

def get_hlines(Hmax, slips):
    """ Get position of slips (useful for plotting)
    position = Hmax-sum(previous slips)
    INPUTS : Hmax, scarp height (without the trenh depth), type (float)
             slips, slip array, type (1D numpy array)
    OUTPUT : h_lines, position for the horizontal lines, type (1D numpy array)
    """
    h_lines=np.zeros((len(slips)))
    for i in range (0, len(slips)):
        h_lines[i]=Hmax-np.sum(slips[0:i])
    return h_lines
    
def plot_min_max(clAMS, height, inferred_cl36, Hscarp, trench_depth, plot_name='min_max', sigAMS=np.array([]), slips=np.array([])):
    
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
    # find max, min and median model
    max_cl36=np.max(inferred_cl36, axis=1)
    min_cl36=np.min(inferred_cl36, axis=1)
    median_cl36=np.median(inferred_cl36, axis=1)
    
    # condition to plot the errorbar on data or not 
    if len(sigAMS)!=0:
        plt.errorbar(clAMS, (height-trench_depth)*1e-2, xerr=sigAMS, marker='.', linestyle='', color='black', label='Measured concentration')
    else:
        plt.plot(clAMS, (height-trench_depth)*1e-2, marker='.', linestyle='', color='black', label='Measured concentration')
    # plot the median model and the greyfill
    plt.plot(median_cl36, (height-trench_depth)*1e-2, color='cornflowerblue', label='Inferred concentration') # plt the median
    plt.fill_betweenx((height-trench_depth)*1e-2, min_cl36, max_cl36, color='cornflowerblue', label='All models', alpha=0.3) # fill between max and min model
    
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
        h_lines=get_hlines(Hscarp, slips)
        plt.hlines(h_lines, low_bound, high_bound, linestyle=':', color='grey')
        
    # labels
    plt.title('Resulting Profile')
    plt.xlabel('[$^{36}$Cl] (at/g)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.savefig(plot_name+'.png', dpi = 1200)
    plt.close('all')
    return

def plot_2D(Var1, Var2, RMSE, x_label, y_label, title, true_values=np.array([]), median_values=np.array([])):
    
    """ Plot of tested values Var1 vs Var2 during inversion, color varying with RMSw
        INPUTS : Var1, variable 1, np.array
                 Var2, variable 2, np.array
                 RMSE, error associated to the model from Var1 and Var2, np.array
                 x_label, x label, string
                 y_label, y label, string
                 title, title of the plot, also used in naming the saved figure, string
                 true_values, true values for Var1, Var2, np.array(Var1_true, Var2_true), default=None
                 median_values, inferred values for Var1, Var2, np.array(Var1_true, Var2_true), default=None
        OUTPUT : scatter plot Var1 vs Var2"""
        
    plt.clf() # clear all figure
    plt.figure(figsize=(8.5, 5.3))
    plt.title(title.replace('_', ' '))
    plt.scatter(Var1, Var2, c=RMSE, cmap='viridis_r', alpha=0.5)
    if len(true_values)!=0:
        plt.plot(true_values[0], true_values[1], marker='*', linestyle='', color='firebrick', label='True value', markersize=12)
        plt.legend()
    if len(median_values)!=0:
        plt.plot(median_values[0], median_values[1], marker='*', linestyle='', color='cornflowerblue', label='Infered value', markeredgecolor='black', markeredgewidth=0.5, markersize=12)  
        plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar(label='Weighted RMSE')
    plt.savefig(title+'.png', dpi=1200)
    plt.close('all')
    return
def scatter_hist(x, y, hist_colors=[], cmap_density='viridis_r', vlines_color='black'):
    
    
    # Start with a square Figure.
    # Draw the scatter plot and marginals.
    fig = plt.figure(figsize=(6, 6))

    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal Axes and the main Axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    # no labels for the histograms
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    number_of_events=np.shape(x)[1]
    cm = matplotlib.colormaps[cmap_density] # define a colormap
    max_bar_heights_x=np.zeros((number_of_events)) # store the max bar height to define the vlines later
    max_bar_heights_y=np.zeros((number_of_events))
    median_x=np.zeros((number_of_events)) # store the median value to define the vlines later
    median_y=np.zeros((number_of_events))
    
    if hist_colors==[]:
        hist_colors = plt.cm.bone(np.linspace(0, 0.7, number_of_events))
    # the scatter plot: 
    for i in range(0, np.shape(x)[1]):
        try:
            xy_kde = np.vstack([x[:,i], y[:,i]])
            z = gaussian_kde(xy_kde)(xy_kde) # first calculate the density of points
            ax.scatter(x[:,i], y[:,i], c=z, cmap=cm, alpha=0.015) # plot with a cmap corresponding to the density of points
        except:
            print('Density could not be estimated, cmap was not applied')
            ax.scatter(x[:,i], y[:,i], c='black', alpha=0.01) # plot with a cmap corresponding to the density of points
    
    # the histograms : in a loop to vary the color of the histogram
    bin_width=(np.max(y)-np.min(y))/100
    bin_width_x=(np.max(x)-np.min(x))/100
    for i in range(0, number_of_events):
        bar_heights_x, bins, patches=ax_histx.hist(x[:,i], bins=np.arange(np.min(x[:,i]),np.max(x[:,i])+bin_width_x, bin_width_x), color=hist_colors[i], alpha=0.8)
        bar_heights_y, bins, patches=ax_histy.hist(y[:,i], bins=np.arange(np.min(y[:,i]),np.max(y[:,i])+bin_width, bin_width), orientation='horizontal', color=hist_colors[i], alpha=0.8)
        max_bar_heights_x[i]=np.max(bar_heights_x)
        max_bar_heights_y[i]=np.max(bar_heights_y)
        median_x[i]=np.median(x[:,i])
        median_y[i]=np.median(y[:,i])
    
    # plot vertical lines corresponding to the median value on the histograms
    ax_histx.vlines(median_x, 0, np.max(max_bar_heights_x), color=vlines_color)
    ax_histy.hlines(median_y, 0, np.max(max_bar_heights_y), color=vlines_color)
    ax.set_xlabel('Ages (yr BP)') # x label
    ax.set_ylabel('Slips (cm)') # y label
    plt.savefig('scatter_plot.png', dpi=1200)
    # plt.close('all')
    return
    
# %% computational functions
def WRMSE(observed_data, modeled_data, incertitude=np.array([])):
    """ Compute the Weighted Root Mean Square Error between two datasets
    
    INPUTS : observed_data, 1D array
             modeled_data, 1D array
             incertitude, 1D array
             
    OUTPUT : rmsw, weighted RMS, float"""
    
    if len(incertitude)==0:
        incertitude=np.ones((len(modeled_data))) # to avoid infinity value when incertitude is not known
    rmsw=np.sqrt(np.mean(((observed_data-modeled_data)/incertitude)**2))
    return rmsw
    
def AICC(measurements, calculations, nb_param):
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
    