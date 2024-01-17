#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:29:54 2024

@author: llinares
"""
import sys
import numpy as np
import os.path

# sys.argv
names=['sf', 'coll', 'data']
non_existing=[]
for i in range(0, len(names)):
    filename=names[i]
    check_file=os.path.isfile(filename+'.txt')
    if check_file==True:
        file_to_convert=np.loadtxt(filename+'.txt')
        np.savetxt(filename+'.csv', file_to_convert, delimiter=',')
    else:
        non_existing.append(filename+'.txt')
        
if len(non_existing)!=0:
    print('Files not found (check spelling!):', non_existing)
else:
    print('All files correctly converted')