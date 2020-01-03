# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:11:42 2019

@author: welch
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

def read_linescan(filename):
    filename = os.getcwd() + '\\' + filename
    print(filename)
    location, te, bi, se = np.loadtxt(filename, delimiter=',',
                                      skiprows = 15,usecols = (1,2,3,4),
                                      unpack = True)
    return location, te, bi, se

def read_spectrum(filename):
    filename = os.getcwd() + '\\' + filename
    print(filename)
    voltage, counts = np.loadtxt(filename, delimiter=',',
                                 skiprows = 15, unpack = True)
    return voltage, counts

def plot(x, y1, y2, y3, name = 'plot'):
    plt.figure()
    plt.title('EDS Linescan Results')
    plt.xlabel('Location (nm)')
    plt.ylabel('Counts')
    plt.plot(x, y1, label='Te', color = 'blue')
    plt.plot(x, y2, label='Bi', color = 'black')
    plt.plot(x, y3, label='Se', color='red')
    plt.legend()
    plt.savefig(name, format = 'svg')
    plt.show()
def plot_spectrum(x, y, name = 'spectrum_plot'):
    peaks, _ = signal.find_peaks(y, height=10000,distance = 10)
    plt.figure()
    plt.title('EDS Spectrum Results')
    plt.xlabel('eV')
    plt.ylabel('Counts')
    plt.plot(x[peaks], y[peaks], 'x', color = 'orange', linewidth = 10)
    plt.plot(x, y)
    plt.show()
    return peaks
def main():
# =============================================================================
#     file2 = '2N_linescan2.csv'
#     os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\
# \Materials_Characterization\\USF_2N\\20191017_EDS')
#     x, te, bi, se = read_linescan(file2)
#     plot(x, te, bi, se, 'linescan2')
# 
#     file3 = '2N_linescan3.csv'
#     os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\
# \Materials_Characterization\\USF_2N\\20191017_EDS')
#     x, te, bi, se = read_linescan(file3)
# 
#     plot(x, te, bi, se, 'linescan3')
# =============================================================================

    file = 'spectrum.csv'
    os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\
\Materials_Characterization\\USF_1N\\USF-1N-HP-S1\\Old_data\\EDS')
    vol, counts = read_spectrum(file)
    peaks = plot_spectrum(vol, counts)
    print(vol[peaks])



if __name__ == '__main__':
    main()