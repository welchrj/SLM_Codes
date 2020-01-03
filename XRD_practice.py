# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:44:55 2019

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


angle_hp, count_hp, angle_slm, count_slm = [], [], [], []
data_hp = file_read(file1)
data_slm = file_read(file2)
data_hp = [x.split(',') for x in data_hp]
data_slm = [x.split(',') for x in data_slm]
for i in range(1,len(data_hp)):
    angle_hp.append(float(data_hp[i][0]))
    count_hp.append(float(data_hp[i][1]))

for i in range(1,len(data_slm)):
    angle_slm.append(float(data_slm[i][0]))
    count_slm.append(float(data_slm[i][1]))


def g(x, A, mu, sigma):
    ''' Function for gaussian profile based on Amplitude, mean, and standard deviation.
    Parameters
    -----------
    x = x-coordinate
    A = intensity
    mu = mean
    sigma = peak width
    '''
    return A/(sigma*math.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def cost(parameters):
    g_0 = parameters[:3]
    g_1 = parameters[3:6]
    return np.sum(np.power(g(x, *g_0) + g(x, *g_1) - y, 2)) / len(x)

def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel',  'LorentxianModel', 'Voigtmodel']:
            model.set_param_hint('sigma',  min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('amplitude', min=1e-6)
            #default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma':  x_range * random.random()
            }
        else:
           raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func: # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


def update_spec_from_peaks(spec, model_indicies, peak_widths=(10, 25), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        print(model_indicie)
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                    'height': y[peak_indicie],
                    'sigma': x_range / len(x) * np.min(peak_widths),
                    'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies
x_1, y_1, x_2, y_2 = [],[],[],[]
for i in range(len(angle_hp)):
    if angle_hp[i] > 25 and angle_hp[i] < 30:
        x_1.append(angle_hp[i])
        y_1.append(count_hp[i])
        x_2.append(angle_slm[i])
        y_2.append(count_slm[i])
    
spec = {
    'x':x_1,
    'y':y_1,
    'model': [
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
    ]
}

peaks_found = update_spec_from_peaks(spec, [0,1,2,3], peak_widths=(15,))
fig, ax = plt.subplots()
ax.scatter(spec['x'], spec['y'], s=4)
for i in peaks_found:
    ax.axvline(x=spec['x'][i], c='black', linestyle='dotted')

model, params = generate_model(spec)
output = model.fit(spec['y'], params, x=spec['x'])
fig, gridspec = output.plot(data_kws={'markersize':  1})

spec = {
    'x':x_2,
    'y':y_2,
    'model': [
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
    ]
}
peaks_found = update_spec_from_peaks(spec, [0,1,2,3], peak_widths=(15,))
fig, ax = plt.subplots()
ax.scatter(spec['x'], spec['y'], s=4)
for i in peaks_found:
    ax.axvline(x=spec['x'][i], c='black', linestyle='dotted')

model, params = generate_model(spec)
output = model.fit(spec['y'], params, x=spec['x'])
fig, gridspec = output.plot(data_kws={'markersize':  1})
plt.savefig('Fit.png', dpi=1047)
