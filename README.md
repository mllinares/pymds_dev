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

**Check out your installation:**

In your terminal, enter the following command :
```
cd tests
conda activate NAME_OF_YOUR_ENVIRONMENT
python3 run_test.py
```
Package description
-------------------
1) Data files\
data.csv : datafile of the rock chemestry\
coll.csv : datafile of the colluvium chemestry\
sf.csv : datafile of the magnetic field factors\
constants.py : dictionary of all constants\
parameters.py : python class containing the site parameters (to be modified for each site!)\
3) Sand alone scripts\
run_forward.py : run only the forward function\
plot_data.py : plot the data\
gen_synthetics.py : generate a synthetic datafile\
invert.py : invert your datafile
4) Dependences\
util folder : contains utlitary functions\
chemestry_scaling.py : module containing functions for the chemestry scaling\
geometric_scaling_factors : module containing functions for the scling associated to the geometry of the scarp

Example on synthetic dataset
----------------------------
1) Modify the parameters.py\
```python
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
        self.trench_depth = 0 # trench depth (cm)
        self.long_term_relief = 500 * 1e2 # cumulative height due to long term history (cm)

        self.data = np.loadtxt('data_out.csv', delimiter=',') # samples chemestry
        self.coll = np.loadtxt('coll.txt') # colluvial wedge chemistry
        self.sf = np.loadtxt('sf.txt') # scaling factors for neutrons and muons reactions

        self.cl36AMS = self.data[:, 64]
        self.sig_cl36AMS = self.data[:,65]       
        self.h  = self.data[:, 62] # position of samples (cm)
        self.Hfinal = max(self.h)
        self.thick = self.data[:, 63]
        self.th2 = (self.thick/2)*self.rho_rock  # 1/2 thickness converted in g.cm-2
        self.Z = (self.Hfinal - self.h)*self.rho_coll
```

