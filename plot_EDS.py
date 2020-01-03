# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:38:46 2019

@author: welch
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def eds_spectrum(file):
    energy, count = np.loadtxt(file, delimiter=',', unpack=True)
    return energy, count

def plot_eds_spectrum(energy1, count1, energy2, count2):
    fig = plt.figure()
    plt.plot(energy1, count1, label='Melt Grown')
    plt.plot(energy2, count2, label='SLM')
    plt.xlabel('Energy [KeV]')
    plt.ylabel('Counts')
    plt.xlim(0,12)
    plt.legend()
    plt.show()
    fig.savefig('EDS.svg', format='svg')

def plot_eds(energy1, count1, energy2, count2):
    fig = plt.figure()
    C = [0.277, 0.525]
    Bi = [1.901, 2.419, 2.525, 2.735, 3.233, 9.417,11.708,10.835,13.022,12.973,15.245,15.581]
    Te = [0.464, 27.467,3.334,3.604,3.768,4.028,4.3,4.569,4.827]
    Se = [11.204, 12.498, 1.204, 1.244, 1.379, 1.419]

    plt.plot(energy1, count1, label='Melt Grown')
    plt.plot(energy2, count2, label='SLM')
# =============================================================================
#     plt.axvline(C[0], ymin = 0, ymax = C[1]/, linewidth = '1', color='blue', label = 'Carbon') #plot(C[0],C[1])
# =============================================================================
    for i in range(len(C)):
        plt.axvline(x = C[i], ymin=0, ymax= 100,
                    linewidth = '0.5', color='blue')
    for i in range(len(Te)):
        plt.axvline(x = Te[i],  ymin=0, ymax=100,
                    linewidth = '0.5', color = 'yellow')
    for i in range(len(Se)):
        plt.axvline(x = Se[i],  ymin=0, ymax=100,
                    linewidth = '0.5', color = 'black')
    for i in range(len(Bi)):
        plt.axvline(x = Bi[i], ymin=0, ymax= 100,
                    linewidth = '0.5', color='red')

    plt.xlabel('Energy [KeV]')
    plt.ylabel('Counts')
    plt.xlim(3,5)
    plt.legend()
    plt.show()

def main():
    file2 = 'C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF_1N\\USF_1N_SLM_Polished\\EDS\\11July_Spectrum.csv'
    file1 = 'C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF_1N\\USF-1N-HP-S1\\24July\\spectrum_24July.csv'

    energy1, count1 = eds_spectrum(file1)
    energy2, count2 = eds_spectrum(file2)

    plot_eds_spectrum(energy1, count1, energy2, count2)
    plot_eds(energy1, count1, energy2, count2)


if __name__ == '__main__':
    main()
# =============================================================================
# fig, ax = plt.subplots()
# ax1 = plt.subplot(311)
# plt.plot(energy1,count1)
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.title('N-type Bismuth Telluride EDS Results', fontsize = 15)
# # share x only
# #ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
# #plt.plot(energy2,count2)
# #makes the tick labels invisible.
# #plt.setp(ax2.get_xticklabels(),visible=False)
# ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
# plt.plot(energy2, count2,'r')
# plt.setp(ax2.get_xticklabels(),visible=True)
# plt.xlim(0,12)
# plt.ylim(0,500000)
# plt.xlabel('Energy [KeV]')
# ax1.text(-.75, 0.5, 'Count', va='center', rotation='vertical',  fontsize=10)
# ax1.text(4.5,500,'Melt grown',va = 'bottom',fontsize=10)
# ax1.fill_betweenx(count1,energy1)
# ax2.fill_betweenx(count2, energy2, color='red')
# ax2.text(4.5,500,'SLM Processed', va = 'bottom', fontsize=10)
# left = 0.125
# right = 0.9
# bottom = 0.1
# top = 0.9
# wspace = 0.2
# hspace = 0.1
# plt.subplots_adjust(left,bottom,right,top,wspace,hspace)
# plt.show()
# fig.savefig('EDS.svg',format = 'svg')
# 
# 
# plt.Figure
# plt.xlim(0,12)
# plt.ylim(0,450000)
# plt.plot(energy2, count2, 'r')
# plt.xlabel('Energy [KeV]')
# plt.ylabel('Counts')
# plt.title('N-type Bismuth Telluride, EDS')
# plt.savefig('EDS(1).svg',format = 'svg')
# 
# =============================================================================
