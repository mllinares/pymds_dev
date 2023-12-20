# pymds_dev
Dev repo of pymds, test for the true repo

Install
-------
Download the package and extract it in your working directory. If your python install does not meet the following requirements, create a virtual environment and install the libraries.

Requirements :
- Python 3.10.9
- Pyro-ppl 1.8.4
- Pyro-api 0.1.2
- Numpy 1.23.5
- Matplotlib 3.7.0
- Scipy 1.1.10
- Ruptures 1.0.6
  
Creating a virtual environment and installing necessary libraries
------------------------------------------------------------------
With Conda/Miniconda: 

From your terminal : enter the following command

```
conda create -n NAME_OF_YOUR_ENVIRONMENT pyro matplotlib numpy scipy ruptures spyder 
```
Spyder is a python IDE, you can use a different one (PyCharm, Visual Studio, ...)

Check out your installation:
----------------------------
In your terminal, go to the downloaded package directory, enter the following commands:
```
cd *PATH TO PACKAGE*
cd tests
conda activate NAME_OF_YOUR_ENVIRONMENT
python3 run_test.py
```
Success message should print on the terminal window

Package description
-------------------
1) Data files\
data.csv : datafile of the rock chemestry\
coll.csv : datafile of the colluvium chemestry\
sf.csv : datafile of the magnetic field factors\
constants.py : dictionary of all constants\
parameters.py : python class containing the site parameters (to be modified for each site!)\
seismic_scenario.py : true seismic scenario, you can use this file to set some paramters of the seismic scenario
2) Stand alone scripts\
run_forward.py : run only the forward function\
plot_data.py : plot the data\
gen_synthetics.py : generate a synthetic datafile\
invert.py : invert your datafile
3) Dependences\
util folder : contains utlitary functions\
chemestry_scaling.py : module containing functions for the chemestry scaling\
geometric_scaling_factors : module containing functions for the scling associated to the geometry of the scarp

Example on synthetic dataset
----------------------------
Usage of a python IDE strongly recommended!
1) Modify the invert.py\
Before running the inversion there are some parameters that you can tune:
In the invert.py file you can modify some of the MCMC parameters like the number of steps during the warm-up and sampling phase, the depth of the probability tree and the target acceptancy probability.

```python
""" MCMC parameters, to be set with CAUTION """
pyro.set_rng_seed(20)
w_step = 0  # number of warmup step
nb_sample = 1000 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.9 # target acceptancy probability (<1)
```
You can also modify the parameters you whish to invert : ages are always inverted, but you can choose to invert the slips associated to each event by setting ```invert_slips``` to ```True```.
Alternatively, you can use the rupture package to find the ruputures by setting ```invert_slips``` to ```False``` and ```use_rpt``` to ```True```. If both ```invert_slips``` and ```use_rpt``` are set to ```False```, then the slips used are the one present in ```seismic_scenario.py```.

You can also invert the long term slip rate by setting ```invert_sr``` to ```True```, if set to ```False```, the slip rate (SR) used is the one entered in ```seismic_scenario.py```.
```python
""" Chose parameters to invert """
invert_slips = False # invert slip array ?
use_rpt = True # use rupture package to find slips
invert_sr = False # invert slip rate ?
invert_quies = False # invert quiescence
```

Be advised that the number of parameters you inverse have an impact on execution time, if you only wish to test the package on your computer, we recommend the following settings:
```python
number_of_events = 3
""" Chose parameters to invert """
invert_slips = False # invert slip array ?
use_rpt = True # use rupture package to find slips
invert_sr = False # invert slip rate ?
invert_quies = False # invert quiescence

""" MCMC parameters, to be set with CAUTION """
pyro.set_rng_seed(20)
w_step = 10  # number of warmup step
nb_sample = 1000 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.9 # target acceptancy probability (<1)

```
This should take approximatly 3h to run on a standard computer (CPU i7-1165G7 @ 2.80GHz × 8). If your specks are lower than those, the algorithm may take longer to complete.

2) Run invert.py
from terminal window (does not require to be in a specific working directory):
```
conda activate NAME_OF_YOUR_ENVIRONMENT
spyder
```
Open "invert.py" from the example folder inside your spyder IDE an click on the play button\
![Capture d’écran du 2023-12-20 16-46-57](https://github.com/mllinares/pymds_dev/assets/126869078/5bf5ee0c-0af3-4ebd-9e2c-f25f729b8f1c)

Or from terminal window inside the "example" directory run :
```
conda activate NAME_OF_YOUR_ENVIRONMENT
nohup python3 invert.py
```
A progress bar indicates the progression of the algorithm\
![Capture d’écran du 2023-12-20 16-40-56](https://github.com/mllinares/pymds_dev/assets/126869078/83afc528-217d-485e-935f-2da3c2bb906f)

When the inversion is done, plots and ".txt" files are generated:

- age.txt : infered ages at each iteration
- infered_cl36.txt : all associated cl36 profile
- RMSw.txt : root mean square associated to the tested cl36 profiles
- sigma.txt : infered sigma
- summary.txt : summary of the inversion
- nohup.out : progression of the algorithm
- slip.txt : infered slip (if inversed)
- SR.txt : infered long term slip rate (if inversed)

If you run the example dataset with the recommanded settings, you should see these 3 plots:
![Event 1](https://github.com/mllinares/pymds_dev/assets/126869078/74c33cb0-ce32-4a39-81f5-154461dd7377)
![Event 2](https://github.com/mllinares/pymds_dev/assets/126869078/e940f081-7f56-453c-8574-f473a814e59c)
![Event 3](https://github.com/mllinares/pymds_dev/assets/126869078/e0052e4d-7070-44fd-8473-f55b21426f87)

Note to new python users
-------------------------
If you are new to python, please refer to the ```quick_python_intro.ipynb```  and to the ```quick_pyro_intro.ipynb``` before going further.\
The notebooks contain useful info on numpy, pyro, pytorch and matplotlib\
Run the folowing from a terminal
```
conda activate NAME_OF_YOUR_ENVIRONMENT
jupyter notebook
```
Open the notebooks from Jupyter interface.

How to use on true dataset
--------------------------
1) Copy and paste your datafiles, in .csv format (delimiter must be ',')
2) Modify the following inside the "parameters.py" file according to your own site:

```python
        self.site_name='9m_numpy_hmax'
        self.rho_rock = 2.66 # rock mean density
        self.rho_coll = 1.5 # colluvium mean density
        self.alpha = 25 # colluvium dip (degrees)
        self.beta = 55 # scarp dip (degrees)
        self.gamma = 35 # eroded scarp dip (degrees)
        self.trench_depth = 0 # trench depth (cm)
        self.long_term_relief = 500 * 1e2 # cumulative height due to long term history (cm)

        self.data = np.loadtxt('data_out.csv', delimiter=',') # samples chemestry
        self.coll = np.loadtxt('coll.csv', delimiter=',') # colluvial wedge chemistry
        self.sf = np.loadtxt('sf.csv', delimiter=',') # scaling factors for neutrons and muons reactions

```
3) Modify seismic_scenario.py :
If you have constraints on the following parameters, fill in the arrays.
If not, do NOT modify this file but keep in mind that more you have constraints (i. e. on the slip rate or pre-exposure) the faster the algorithm
```python
seismic_scenario['ages'] = np.array([]) # exhumation ages, older to younger (yr)
seismic_scenario['slips'] = np.array([]) # slip corresponding to the events (cm)
seismic_scenario['SR'] = 0 # long term slip rate of your fault (mm/yr)
seismic_scenario['preexp'] = 0 # Pre-exposition period (yr)
seismic_scenario['quiescence'] = 0 # Quiescence period (yr)
```

4) Modify MCMC parameters in the invert.py script
```python
""" MCMC parameters, to be set with CAUTION """
pyro.set_rng_seed(20)
w_step = 0  # number of warmup (~30% of total models)
nb_sample = 1000 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.9 # target acceptancy probability (<1)
```
It is recommended to first test the algorithm without the warmup, low number of samples, low tree depth and low target acceptancy probaility and then increase those.
As discussed in the paper, the algorithm performs better with low warm_up steps.
If you have no idea of what those are, it is recommended that you download the jupyter notebook ```quick_pyro_intro.ipynb``` before using pymds on true datasets.

Link to previous publications
-----------------------------
github

License
-------

