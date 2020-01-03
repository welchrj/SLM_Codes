#!/usr/bin/env python
# coding: utf-8


import numpy
from matplotlib import pyplot



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

def laser_scan(X, Y, g, v, n, dt):
    ds = v*dt*n
    Z = numpy.sqrt((X)**2 + (Y-ds)**2) - .025

def ftcs(X, Y, T0, nt, dt, dx, dy, alpha, beta):
    """
    Computes and returns the temperature distribution
    after a given number of time steps.
    Explicit integrations using forward differencing
    in time and central differencing in space, with
    Neumann conditions on the boarders
    
    Parameters
    ----------
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
    
    T = T0.copy()
    ny, nx = T.shape
    I, J = int(nx/2), int(ny/2)
    for n in range(nt):
        T[1:-1,1:-1] = (T[1:-1,1:-1] +
                        sigma_x * (T[1:-1,2:] - 2.0 * T[1:-1,1:-1] + T[1:-1,:-2]) +
                        sigma_y * (T[2:,1:-1] - 2.0 * T[1:-1,1:-1] + T[:-2,1:-1])+ 
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
        
    return T


def main():
    # Stability condition sigma
    sigma = 0.25
    dt = sigma * min(dx,dy)**2 / D
    nt = 5
    print('Total Time: {}'.format(nt*dt))
    
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
    # movement
    ds = v*dt
    
    pyplot.figure()
    contf = pyplot.contourf(x,y,Z)
    cbar = pyplot.colorbar(contf)
    
    mask = numpy.where(Z <= 0)
    G[mask] = g
    
    T = ftcs(T0,nt,dt,dx,dy,D,E,G)
    
    
    
    
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




