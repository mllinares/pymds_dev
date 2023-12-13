# pymds_dev
Dev repo of pymds (hidden) 

Install
-------
Just download the package, extract it in your working directory.
- Requirements :
- Python 3.10.9
- Pyro-ppl 1.8.4
- Pyro-api 0.1.2
- Numpy 1.23.5
- Matplotlib 3.7.0
- Scipy 1.1.10
- Ruptures 1.0.6
  
Creating a virtual environment to install those libraries is highly recommended.

With Conda/Miniconda: 

From your terminal : enter the following command

```
conda create -n NAME_OF_YOUR_ENVIRONMENT pyro matplotlib numpy scipy ruptures spyder 
```
Spyder is a python IDE, you can use a different one (PyCharm, Visual Studio, ...)

Check out your installation:
----------------------------
In your terminal, enter the following command, go to the package directory:
```
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
2) Sand alone scripts\
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
Usage of a python IDE strongly recommended!\
from terminal window (does not require to be in a specific working directory):
```
conda activate NAME_OF_YOUR_ENVIRONMENT
spyder
```
1) Run invert.py
Open "invert.py" from the example folder inside your IDE an click on the play button\
Or from terminal window inside the "example" directory run :
```
nohup python3 invert.py
```
A progress bar indicates the progression of the algorithm\
When the inversion is done, plots and ".txt" files are generated
-age.txt : tested ages at each iteration
-cl_36_infered.txt : all associated cl36 profile
-summary.txt : summary of the inversion
2) You can modify some MCMC parameters

```python
""" MCMC parameters, to be set with CAUTION """
pyro.set_rng_seed(20)
w_step = 0  # number of warmup (~30% of total models)
nb_sample = 1000 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.9 # target acceptancy probability (<1)
```
Note to new python users
-------------------------
If you are new to python, please refer to the "quick_python_intro.ipynb"  and to the "quick_pyro_intro.ipynb" before going further.\
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
It is recommended to first test the algorithm without the warmup, low number of smaples, low tree depth and low target acceptancy probaility and then increase those.
As discussed in the paper, the algorithm performs better with low warm_up steps.
If you have no idea of what those are, it is recommended that you download the jupyter notebook "quick_pyro_intro.ipynb" before using pymds on true datasets.
