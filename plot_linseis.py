# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:47:01 2019
Plot .asc file from Linseis Seebeck data.
@author: welch
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF_1N\\USF_1N_SLM_Polished\\Linseis\\LSR_Data')
global data
data = []
''' Uncertainty
    1. Average value
    2. deviation of each measurement
    3. average of each deviation '''
# =============================================================================
# file1 = 'Bi2Te3-USF1aN-SLM_20190313.asc'
# data1 = np.loadtxt(file1, skiprows=51, unpack = True)
# =============================================================================
def read_all():
    with os.scandir(os.getcwd()) as it:
        for entry in it:
            if entry.name.endswith('.ASC') and entry.is_file():
                print(entry.name)
                data.append(np.loadtxt(entry.name, skiprows=51, unpack=True))

def averages(all_data, col):

    temp = []
    size = len(data[0][0])
    for i in range(size):
        add = 0
        for j in range(len(data)):
            add = add + data[j][col][i]
        temp.append(add)

    for i in range(len(temp)):
        temp[i] = temp[i]/len(data)
    deviations = []
    for i in range(size):
        dev = []
 
        for j in range(len(data)):
            dev.append(abs(data[j][col][i] - temp[i]))
        deviations.append(np.average(dev))

    return temp, deviations

def plot_all(dater):
    '''Plots all tables in each file on single figure
    Parameters
    ----------
    dater = array, with all data read.
    
    Returns
    -------
    None'''
    plt.figure
    for i in range(len(dater)):
        plt.plot(dater[i][1], dater[i][4])

    plt.show()
    plt.figure
    for i in range(len(dater)):
        plt.plot(dater[i][1], dater[i][3])

    return None

def conductivity(data, col):
    temp = []
    size = len(data[0][0])
    for i in range(size):
        add = 0
        for j in range(len(data)):
            add = add + (1/data[j][col][i])
        temp.append(add)

    for i in range(len(temp)):
        temp[i] = temp[i]/len(data)
    deviations = []
    for i in range(size):
        dev = []
 
        for j in range(len(data)):
            dev.append(abs(1/data[j][col][i] - temp[i]))
        deviations.append(np.average(dev))
    

    return temp, deviations
    
headers = ['Time','Temperature','Temperature gradient','Resistivity','Relative seebeck coefficient','Current','Voltage']

def main():
    read_all()
    avg_temp, err_temp = averages(data, 1)
    avg_see, err_see = averages(data, 4)
    avg_con, err_con = conductivity(data, 3)
    x, y, y2, errx, erry, erry2 = [],[],[],[],[],[]
    print(avg_temp)
    print(avg_see)
    print(avg_con)
    for i in range(len(avg_temp)):
        x.append(avg_temp[i])
        y.append(avg_see[i])
        y2.append(avg_con[i])
        errx.append(err_temp[i])
        erry.append(err_see[i])
        erry2.append(err_con[i])
        if avg_temp[i] > avg_temp[i+1]:
            break;
    
    fig, ax1 = plt.subplots()
    #color = 'tab:black'
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Seebeck coefficient (µV/K)')
    ax1.plot(x, y, '-k')
    #ax1.tick_params(axis='y', labelcolor = color)
    
    ax2 = ax1.twinx() #second axis that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Conductivity (µS/m)')
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.errorbar(x, y2, yerr = erry2, fmt='.r', elinewidth=1, capsize=2)
    ax1.errorbar(x, y, yerr = erry, xerr = errx, fmt='.k', elinewidth=1 , capsize=2)
    plt.title('Seebeck Coefficient and Electrical Conductivity')
    plt.show()
    fig.savefig('seebeck.svg', format = 'svg')

    plot_all(data)
if __name__ == '__main__':
    main()

