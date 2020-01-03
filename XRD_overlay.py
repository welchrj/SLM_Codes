# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:44:52 2019

@author: welch
"""


import os
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize,signal

from lmfit import models
os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF_2N\\XRD\\XRD_7_19_2019')
file1 = 'USF-2N-HP-51B_07_19_19.txt'
file2 = 'USF-2N-SLM-51B_07_19_19.txt'

def file_read(file):
    data = []
    with open(file,'r') as f:
        for line in f:
            if '[Data]' in line:
                for line in f:
                    data.append(line)
    return data

def split_data(dat):
    angle, count = [],[]
    
    for i in range(1,len(dat)):
        angle.append(float(dat[i][0]))
        count.append(float(dat[i][1]))
    
    return angle, count

def overlay(x1,y1,x2,y2):
    plt.figure(1)
    plt.plot(x1,y1,'k',linewidth=0.5)
    plt.plot(x2,y2,'r',linewidth=0.5)
    plt.title('XRD')
    plt.xlabel('2theta')
    plt.ylabel('Intensity')
    plt.savefig('XRD.svg',format = 'svg')
    
def main():
    data1 = file_read(file1)
    data2 = file_read(file2)
    data_hp = [x.split(',') for x in data1]
    data_slm = [x.split(',') for x in data2]
    
    x_hp, y_hp = split_data(data_hp)
    x_slm, y_slm = split_data(data_slm)
    
    overlay(x_hp,y_hp,x_slm,y_slm)
if __name__ == '__main__':
    main()