# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:01:28 2019

@author: welch
"""

from scipy import stats
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
import os
#os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF-1N_powder\\7_august')
#N, area, circ, feret, feretX, feretY, feretang, minferet, ar, roundness, soidity = np.loadtxt(file1,delimiter=',',skiprows=1,unpack=True)

#Plots particle distribution, Feret's diameter
# =============================================================================
# os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF-1N_powder\\7_august')
# file3 = 'Results_particle_distribution_1200X.csv'
# 
# N, area, circ, feret, feretX, feretY, feretang3, minferet, ar, roundness, length = np.loadtxt(file3,delimiter=',',skiprows=1,unpack=True)
# fig = plt.figure()
# plt.hist(circ, density=True,bins=100,histtype='step',color='black')
# xt = plt.xticks()[0]
# xmin, xmax = min(xt), max(xt)
# lnspc = np.linspace(xmin, xmax, len(circ))
# 
# bg, hg  = stats.norm.fit(circ)
# pdf_g = stats.norm.pdf(lnspc,bg,hg)
# plt.plot(lnspc, pdf_g, label="Normal Dist",color='red')
# plt.xlabel('Circularity Index')
# plt.ylabel('Probability Density')
# plt.legend()
# fig.savefig('circularity.svg')
# =============================================================================

# Plots Feret angle of melt grown sample grains
# =============================================================================
# os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF-1N-HP-S1\\7August')
# file2 = 'areas.csv'
# 
# N, area, circ, feret, feretX, feretY, feretang2, minferet, ar, roundness, length = np.loadtxt(file2,delimiter=',',skiprows=1,unpack=True)
# fig = plt.figure()
# plt.hist(feretang2,density=True, bins=25)
# xt = plt.xticks()[0]
# xmin, xmax = min(xt), max(xt)
# lnspc = np.linspace(xmin, xmax, len(feretang2))
# plt.xlabel('Feret Angle°')
# plt.ylabel('Probability Density')
# fig.savefig('grain_orientation.svg')
# 
# =============================================================================

# Plots feret Angle of SLM sample
# =============================================================================
# os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF_2N\\USF_2N_SLM_S1A\\Leica\\8august')
# file3 = 'grain_areas(1).csv'
# 
# N, area, circ, feret, feretX, feretY, feretang3, minferet, ar, roundness, length = np.loadtxt(file3,delimiter=',',skiprows=1,unpack=True)
# fig = plt.figure()
# plt.hist(feretang3, density=True, bins=25)
# xt = plt.xticks()[0]
# xmin, xmax = min(xt), max(xt)
# lnspc = np.linspace(xmin, xmax, len(feretang3))
# 
# g, s  = stats.norm.fit(feretang3)
# pdf_g = stats.norm.pdf(lnspc,g,s)
# plt.plot(lnspc, pdf_g, label="Norm")
# # =============================================================================
# # mu = (np.average(ar)+np.std(ar))/2
# # dist= poisson(mu)
# # x = np.arange(1,16)
# # plt.plot(x,dist.pmf(x),color='red')
# # =============================================================================
# plt.xlabel('Angle from build direction°')
# plt.ylabel('Probability Density')
# fig.savefig('grain_orientation_angle_poisson.svg')
# =============================================================================


# =============================================================================
# # Plots aspect ratio of SLM sample
# os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF_2N\\USF_2N_SLM_S1A\\Leica\\8august')
# file3 = 'grain_areas(1).csv'
# 
# N, area, circ, feret, feretX, feretY, feretang3, minferet, ar, roundness, length = np.loadtxt(file3,delimiter=',',skiprows=1,unpack=True)
# fig = plt.figure()
# plt.hist(ar, density=True, bins=25)
# xt = plt.xticks()[0]
# xmin, xmax = min(xt), max(xt)
# lnspc = np.linspace(1.5, xmax, len(ar))
# 
# ag, bg, cg  = stats.gamma.fit(ar)
# pdf_g = stats.gamma.pdf(lnspc,ag,bg,cg)
# plt.plot(lnspc, pdf_g, label="Norm")
# # =============================================================================
# # mu = (np.average(ar)+np.std(ar))/2
# # dist= poisson(mu)
# # x = np.arange(1,16)
# # plt.plot(x,dist.pmf(x),color='red')
# # =============================================================================
# plt.xlabel('Aspect Ratio')
# plt.ylabel('Probability Density')
# 
# fig.savefig('aspect_ratio(2).svg')
# =============================================================================

# Plots aspect ratio of HP sample
# =============================================================================
# os.chdir('C:\\Users\\welch\\Documents\\GWU\\Research\\Materials_Characterization\\USF-1N-HP-S1\\7August')
# file2 = 'areas.csv'
# 
# N, area, circ, feret, feretX, feretY, feretang2, minferet, ar, roundness, length = np.loadtxt(file2,delimiter=',',skiprows=1,unpack=True)
# fig = plt.figure()
# plt.hist(ar, density=True,bins=25)
# xt = plt.xticks()[0]
# xmin, xmax = min(xt), max(xt)
# lnspc = np.linspace(1,xmax,len(ar))
# ag, bg, cg  = stats.gamma.fit(ar)
# pdf_g = stats.gamma.pdf(lnspc,ag,bg,cg)
# plt.plot(lnspc, pdf_g, label="Norm")
# plt.xlabel('Aspect Ratio')
# plt.ylabel('Probability Density')
# fig.savefig('aspect_ratio_hp.svg')
# =============================================================================
# =============================================================================
# # create some normal random noisy data
# ser = 50*np.random.rand() * np.random.normal(10, 10, 100) + 20
# 
# # plot normed histogram
# plt.hist(ser, normed=True)
# 
# # find minimum and maximum of xticks, so we know
# # where we should compute theoretical distribution
# xt = plt.xticks()[0]  
# xmin, xmax = min(xt), max(xt)  
# lnspc = np.linspace(xmin, xmax, len(ser))
# 
# # lets try the normal distribution first
# m, s = stats.norm.fit(ser) # get mean and standard deviation  
# pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
# plt.plot(lnspc, pdf_g, label="Norm") # plot it
# 
# # exactly same as above
# ag,bg,cg = stats.gamma.fit(ser)  
# pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)  
# plt.plot(lnspc, pdf_gamma, label="Gamma")
# 
# # guess what :) 
# ab,bb,cb,db = stats.beta.fit(ser)  
# pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)  
# plt.plot(lnspc, pdf_beta, label="Beta")
# 
# plt.show()
# =============================================================================
