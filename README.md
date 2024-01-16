# pymds_dev
Dev repo of pymds, test for the true repo

Install
-------
1) Download and extract the package

2) Install Anaconda and required libraries
   
   Anaconda is required to install the libraries, link to Anaconda
   ```
   https://www.anaconda.com/download
   ```
4) Create a virtual environment and install necessary libraries

On Linux/MAC enter the following commands in a terminal, on Windows 10/11 enter the commands in Anaconda prompt terminal

```
conda create -n NAME_OF_YOUR_ENVIRONMENT matplotlib numpy scipy git jupyter spyder
conda activate NAME_OF_YOUR_ENVIRONMENT
conda install conda-forge::pyro-ppl
conda install conda-forge::ruptures
```
Note : Spyder is a python IDE and its installation is not required you can use a different one (PyCharm, Visual Studio, ...). Jupyter is required for the tutorial.

To intall the devellopment version of pyro (required to use RandomWalk kernel, see section NUTS or RandomWalk):
```
pip install git+https://github.com/pyro-ppl/pyro.git
```

Check out your installation
----------------------------
From a terminal window (or Anaconda prompt), go to the downloaded package directory ```cd *PATH TO PACKAGE*``` and enter the following commands:
```
conda activate NAME_OF_YOUR_ENVIRONMENT
python3 test_install.py 
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
Usage of a python IDE strongly recommended!\
An Example folder is provided with synthetic dataset corresponding to the following scenario : 3 earthquakes (7000, 2500, 500 BCE), each generating 300 cm of displacement, slip rate of 0.3 mm/yr\
Note : You can generate your own synthetic files with the ``` generate_synthetics.py``` script.

You can run the inversion either from python IDE or through command lines:

a) With Spyder

From terminal window (does not require to be in a specific working directory):
```
conda activate NAME_OF_YOUR_ENVIRONMENT
spyder
```
Open "invert.py" from the example folder inside your spyder IDE an click on the play button


b) With command lines

From terminal window inside the "example" directory :
```
conda activate NAME_OF_YOUR_ENVIRONMENT
nohup python3 invert.py
```
This should take approximatly 3h to run on a standard home computer (CPU intel i7-1165G7, 2.80GHz, RAM 16Go). If your specks are lower than those, the algorithm may take longer to complete.

In this example, only the ages are searched and you should generate the same following plots.


Inside the example folder, you can find a folder called ```expected_results``` wich contains the expected plots and result files

You can notice some parameters that you can tune like the number of steps during the warm-up and sampling phase, the depth of the probability tree and the target acceptancy probability. All have an in impact on the efficeincy of the parameter search. The important thing is that the tree depth and warm up can be set to low values to cut the runtime without impacting the results. You can test the impact of modifying those on a simple linear function (see jupyter notebook ```quick_pyro_intro.ipynb```).

```python
""" MCMC parameters, to be set with CAUTION """
pyro.set_rng_seed(20)
w_step = 10  # number of warmup step
nb_sample = 4000 # number of samples
tree_depth = 1 # maximum probability tree depth (min: 4, max: 10) 
target_prob = 0.9 # target acceptancy probability (<1)
```
You can also modify the parameters you whish to invert : ages are always inverted, but you can choose to invert the slips associated to each event by setting ```invert_slips = True```.
Alternatively, you can use the rupture package to find the ruptures by setting ```invert_slips = False``` and ```use_rpt = True```. If both ```invert_slips``` and ```use_rpt``` are set to ```False```, then the slips used are the one present in ```seismic_scenario.py```.\
You can also invert the long term slip rate by setting ```invert_sr = True```, if set to ```False```, the slip rate (SR) used is the one entered in ```seismic_scenario.py```.\
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
If not, do NOT modify this file but keep in mind that more you have constraints (i. e. on the slip, slip rate or pre-exposure) the faster the algorithm
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
As discussed in the paper, the algorithm performs better with low warm_up steps and low tree depth.
If you have no idea of what those are, it is recommended that you download the jupyter notebook ```quick_pyro_intro.ipynb``` before using pymds on true datasets.

Link to previous publications
-----------------------------
github

License
-------

