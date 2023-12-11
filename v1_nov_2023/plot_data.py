#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 09:24:26 2023

@author: llinares
"""

import numpy as np
import matplotlib.pyplot as plt
import parameters
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

name='9m_hmax'
param = parameters.param() # import site parameters
height = param.h
sig=param.sig_cl36AMS
cl36=param.cl36AMS

fig=plt.figure(num=1, dpi=1200, figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(cl36, height, xerr=sig, color='black', marker='.', linestyle='')
plt.xlabel('cl36 (at/g)')
plt.ylabel('h (cm)')

# Major ticks every 20, minor ticks every 5
# Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(5*1e4))
# ax.yaxis.set_major_locator(MultipleLocator(100))

# Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(4))
# ax.yaxis.set_minor_locator(AutoMinorLocator(4))

# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)