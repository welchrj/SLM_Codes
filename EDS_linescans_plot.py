# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:57:29 2019

@author: welch
"""


import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF_1N\\USF_SLM_1N_S1\\EDS\\')
file1 = 'Line_scan_1.csv'
file2 = 'Line_scan_2.csv'

point1, distance1, te1, bi1, se1, image1, frame1 = np.loadtxt(file1, delimiter=',',dtype='str',skiprows=15,unpack=True)
point2, distance2, te2, bi2, se2, image2,a, b,c = np.loadtxt(file2, delimiter=',',dtype='str',skiprows=15, unpack=True)

for i in range(len(distance1)):
    distance1[i] = float(distance1[i])
    te1[i] = float(te1[i])
    bi1[i] = float(bi1[i])
    se1[i] = float(se1[i])

for i in range(len(distance2)):
    distance2[i] = float(distance2[i])
    te2[i] = float(te2[i])
    bi2[i] = float(bi2[i])
    se2[i] = float(se2[i])

xmin = min(distance1)
xmax = max(distance1)
plt.figure()
lnspc = np.linspace(xmin, xmax)
plt.plot(lnspc, distance1, te1)
