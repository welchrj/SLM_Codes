# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:57:49 2019

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
    
x = np.linspace(-3, 3, 1000)
fig, ax = plt.subplots()
ax.plot(x, g(x, 1, 0, 1))
plt.show()

def f(x):
    '''A random function used for an example'''
    return np.exp(-(x-2)**2) + np.exp(-(x-6)**2/10) + 1/(x**2 + 1)


'''Use scipy.optimize.minimize to find global minimum'''

print('|{:16}|{:>16}|{:>16}|'.format('initial','iterations','minimum'))
initial_guess = -0.5
result = optimize.minimize(lambda x: -f(x), [initial_guess])
print(f'|{initial_guess:+16.1f}|{result.nit:>16}|{result.x[0]:16.3f}|')


A = 100
mu = 4.0
sigma = 4.0
n = 200
x = np.linspace(-10,10,n)
y = g(x, A, mu, sigma) + np.random.randn(n)

def cost(parameters):
    '''Function cost, calculates the mean square error between the 
    gaussian fit g(), and the function y

    Parameters
    ---------
    a, b, c = parameters
    #y has been calculated in previous snippet
    
    return np.sum(np.power(g(x, a, b, c) - y, 2)) / len(x)
    
result = optimize.minimize(cost, [0, 0, 1])
print('steps', result.nit, result.fun)
print(f'amplitude: {result.x[0]:3.3f} mean: {result.x[1]:3.3f} sigma: {result.x[2]:3.3f}')
fig, ax = plt.subplots()
ax.scatter(x, y,s=2)
ax.plot(x, g(x, *result.x))
plt.show()

#Fit two gaussians
g_0 = [250.0, 4.0,  5.0]
g_1 = [20.0, -5.0, 1.0]
n = 150
x = np.linspace(-10, 10, n)
y = g(x, *g_0) + g(x, *g_1) + np.random.randn(n)

fig, ax = plt.subplots()
ax.scatter(x,y,s=1)
ax.plot(x, g(x, *g_0))
ax.plot(x, g(x, *g_1))
plt.show()

def cost(parameters):
    g_0 = parameters[:3]
    g_1 = parameters[3:6]
    return np.sum(np.power(g(x, *g_0) + g(x, *g_1) - y, 2)) / len(x)

initial_guess = [1, 0, 1, -1, 0 , 1]
result = optimize.minimize(cost, initial_guess)
print('steps', result.nit,  result.fun)
print(f'g_0: amplitude: {result.x[0]:3.3f} mean: {result.x[1]:3.3f} sigma: {result.x[2]:3.3f}')
print(f'g_1: amplitude: {result.x[3]:3.3f} mean: {result.x[4]:3.3f} sigma: {result.x[5]:3.3f}')

fig, ax = plt.subplots()
ax.scatter(x,y,s=1)
ax.plot(x, g(x, *result.x[:3]))
ax.plot(x, g(x, *result.x[3:6]))
ax.plot(x, (g(x, *result.x[:3]) + g(x, *result.x[3:6])))
plt.show()

# Using lmfit

model_1 = models.GaussianModel(prefix='m1')
model_2 = models.GaussianModel(prefix='m2')
model = model_1 + model_2

params_1 = model_1.make_params(center=1, sigma=1)
params_2 = model_2.make_params(center=-1, sigma=1)
params = params_1.update(params_2)

output = model.fit(y, params, x=x)
fig, gridspec = output.plot(data_kws={'markersize': 1})
plt.show()

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
            raise NotImplementedError(f'model {basis_func["type"]} not implemented yet')
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

spec = {
        'x': x,
        'y': y,
        'model': [
                {'type': 'GaussianModel'},
                {'type': 'GaussianModel'}
            ]
        }

model, params = generate_model(spec)
output = model.fit(spec['y'],  params, x=spec['x'])
fig, gridspec = output.plot(data_kws={'markersize' : 1})
plt.show()

def update_spec_from_peaks(spec, model_indices, peak_widths=(10, 25), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indices, _ = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indices)
    for peak_indices, model_indices in zip(peak_indices.tolist(), model_indices):
        model = spec['model'][model_indices]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                    'height': v[peak_indices],
                    'sigma': x_range / len(x) * np.min(peak_widths),
                    'center': x[peak_indices]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplementedError(f'model {basis_func["type"]} not implemeented yet')
    return peak_indices
        