#!/usr/bin/env python
# coding: utf-8
""" 2D heat diffusion for simulating laser powder bed fusion. 
This program uses an explicit scheme. """

import numpy
from matplotlib import pyplot
import ipywidgets



rho = 7.7e-6  # kg/mm^3
c = 1.4       # J/kgK
k = 2.0/1000  # W/mmK
D = k/(rho*c) # Thermal Diffusitivity
E = 1/(rho*c) 
P = 25        # Watts
d  = .050     # mm
t = .150      # mm
Lx = 1.0
Ly = 1.0
nx = 201
ny = 201
x = numpy.linspace(0.0,Lx,nx)
y = numpy.linspace(0.0,Ly,ny)
dx = Lx/(nx-1)
dy = Ly/(ny-1)




# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16



def generation(P,d,t):
    g = P/(numpy.pi*d**2*t) #[W/mm^3]
    return g

def laser_scan(X, Y, v, g, n, dt):
    """ 
    Computes laser scan pattern along the y-direction
    function is called within a for loop to and advanced based
    on scan speed and total time passed.
    
    Parameters
    ----------
    X : numpy.ndarray (2D)
    Y : numpy.ndarray (2D)
    g : float
        Volumetric heat generation
    v : float
        Scan speed
    n : int
        Iteration number
    dt: float
        Time step size

    Returns
    -------
    G : numpy.ndarray
        2D array for heat generated on the powder bed.

    """

    nx, ny = len(X), len(Y)
    ds = v*dt*n
    Z = numpy.sqrt((X-(Lx/2))**2 + (Y-((Ly/2)+ds))**2) - .025
    G = numpy.zeros((nx,ny))
    mask = numpy.where(Z <= 0)
    G[mask] = g

    return G

def ftcs(X, Y, v, g, T0, nt, dt, dx, dy, alpha, beta):
    """
    Computes and returns the temperature distribution
    after a given number of time steps.
    Explicit integrations using forward differencing
    in time and central differencing in space, with
    Neumann conditions on the boarders
    
    Parameters
    ----------
    X  : 2D array
        From numpy.meshgrid(x,y)
    Y  : 2D array
        From numpy.meshgrid(x,y)
    v  : float
        scan speed (mm/s)
    g  : float
        Volumetric heat generation (W/mm^3)
    T0 : numpy.ndarray
        The initial temperature distribution as a 2D array of floats.
    nt : integer
        Maximum number of time steps to compute.
    dt : float
        Time-step size.
    dx : float
        Grid spacing in the x direction.
    alpha : float thermal diffusivity
    
    Returns
    -------
    T : numpy.ndarray
        The temperature distribution as a 2D array of floats"""
    
    sigma_x = alpha*dt / dx**2
    sigma_y = alpha*dt / dy**2
    
    T = numpy.copy(T0)
    T_hist = []
    ny, nx = T.shape
    I, J = int(nx/2), int(ny/2)
    for n in range(nt):
        Tn = numpy.copy(T)
        G = laser_scan(X, Y, v, g, n, dt)
        if n > 500:
            G = numpy.zeros((nx,ny))

        T[1:-1,1:-1] = (Tn[1:-1,1:-1] +
                        sigma_x * (Tn[1:-1,2:] - 2.0 * Tn[1:-1,1:-1] + Tn[1:-1,:-2]) +
                        sigma_y * (Tn[2:,1:-1] - 2.0 * Tn[1:-1,1:-1] + Tn[:-2,1:-1])+
                        dt*beta*G[1:-1,1:-1])
        # Apply Neumann conditions on the boundaries
        T[-1,:] = T[-2,:]
        T[0,:]  = T[1,:]
        T[:,-1] = T[:,-2]
        T[:,0]  = T[:,1]
#         print(len(T))
#         if T[J,I] >= 500+273.15:
#             break
#         print('[time step {}] Center at T={:.2f} at t={:.2f} s'
#               .format(n+1,T[J,I], (n+1))*dt)
        T_hist.append(T)

    return T, T_hist

def main():

    # Stability condition sigma
    sigma = 0.25
    dt = sigma * min(dx,dy)**2 / D
    nt = 5
    print('Total Time: {}'.format(nt*dt))
    print('Time step sie: {}'.format(dt))

    # Set initial temperature
    T0 = 20.0 * numpy.ones((ny,nx))
    
    # Initial condition for laser at the center
    G = numpy.zeros((nx,ny))
    g = generation(P,d,t)
    X, Y = numpy.meshgrid(x,y)

    # Center
    # Z = (X - (Lx/2))**2 + (Y - (Ly/2))**2 - .025**2
    # Start laser at origin
    # Z = X**2 + Y**2 - 0.025**2
    # Bottom Center
    Z = (X-(Lx/2))**2 + Y**2 - .025**2
    v = 500 #mm/s
    pyplot.figure()
    contf = pyplot.contourf(X,Y,Z)
    cbar = pyplot.colorbar(contf)
    
    mask = numpy.where(Z <= 0)
    G[mask] = g
    pyplot.figure()
    pyplot.contourf(X,Y,G)

    T, T_hist = ftcs(X, Y, v, g, T0, nt, dt, dx, dy, D, E)
    print(T[int(nx/2), int(ny/2)])


    
    pyplot.figure(figsize=(6.0,6.0))
    pyplot.xlabel('x [mm]')
    pyplot.ylabel('y [mm]')
    levels = numpy.linspace(20.0,400, num=51)
    contf = pyplot.contourf(x, y, T, levels=levels)
    cbar = pyplot.colorbar(contf)
    cbar.set_label('Temperature [C]')
    pyplot.axis('scaled', adjustable='box')

if __name__ == '__main__':
    main()




