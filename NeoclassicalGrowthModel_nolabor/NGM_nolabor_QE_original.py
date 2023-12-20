#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Computational imports
import numpy as np
from scipy import optimize as opt
from numba import njit

# Graphics imports
import matplotlib.pyplot as plt
import seaborn as sns  # Better quality figures
from matplotlib import rcParams
rcParams['figure.figsize'] = (9, 6)  # Sets the size of the figures in the notebook
from matplotlib import cm # for 3d poltting
from mpl_toolkits.mplot3d.axes3d import Axes3D # for 3d poltting

@njit
def c_poly(k,z,η):
    """
    Returns a polynomial approximation of consumption as a function of the state (k,z)
    """
    
    c_log = η[0] + η[1]*np.log(k) + η[2]*np.log(z) + η[3]*np.log(k)**2 + η[4]*np.log(z)**2 + η[5]*np.log(k)*np.log(z)
    
    return np.exp(c_log)

n_q = 5  # Number of nodes and weights for the Gauss-Hermite quadrature

# Use the hermgauss function to get the nodes and the weights for the Gauss-Hermite quadrature
gh_quad = np.polynomial.hermite.hermgauss(n_q)

@njit
def euler_err(η,quad):
    """
    Returns the sum of squared Euler errors at all grid points
    
    """
    
    q_nodes, q_weights = quad
    β = 0.99
    α = 0.33
    δ = 0.025
    γ = 1
    ρ = 0.95
    σ = 0.1
    ssr      =  0  # Initialize the sum of squared errors
    
    for i_k in range(len(k_grid)):  # Iterate over k and z grids
        
        for i_z in range(len(z_grid)):
            
            k       = k_grid[i_k]
            z       = z_grid[i_z]
            c       = c_poly(k,z,η)
            k_prime = z * k**α + (1-δ) * k - c;
            
            # Calculating the expectation over the GH nodes for every (k,z) weighted by the GH weights
            # We use the Gauss-Hermite formula with a change of variable
            
            E  = 0
            
            for i_q in range(len(q_nodes)):
                
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]         # The errors are normally distributed with mean 0 and std σ
                z_prime = np.exp(ρ * np.log(z) + e_prime)
                c_prime = c_poly(k_prime,z_prime,η)
                
                E += q_weights[i_q] * β * c_prime**(-γ) * (α * z_prime * k_prime**(α-1) + (1-δ))            
                
            E = E / np.sqrt(np.pi)      
            ssr += (E - c**(-γ))**2

    return ssr

β = 0.99
α = 0.33
δ = 0.025
γ = 4
ρ = 0.95
σ = 0.1

# Calculate the steady state level of capital
k_ss = (β * α/(1-β*(1-δ)))**(1/(1-α))

# Setting up the capital grid
k_low    =  0.5 * k_ss
k_high   =  1.5 * k_ss
n_k =  10
k_grid = np.linspace(k_low,k_high,n_k)

# Setting up the productivity grid (3 std)
z_low    = -3 * np.sqrt(σ**2/(1-ρ**2))
z_high   =  3 * np.sqrt(σ**2/(1-ρ**2))
n_z =  10 
z_grid  = np.exp(np.linspace(z_low,z_high,n_z))

# Set initial values for the coefficients
η_init = np.zeros((1, 6))

# Find solution by minimizing the errors on the grid
η_opt = opt.minimize(euler_err,η_init, args = (gh_quad,) ,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xatol': 1e-10,'fatol': 1e-10}).x
print(η_opt)


k_grid_fine = np.linspace(k_low,k_high,100)
z_grid_fine = np.exp(np.linspace(z_low,z_high,100))

# Generate meshgrid coordinates for 3d plot
kg, zg = np.meshgrid(k_grid_fine, z_grid_fine)

# Plot policy function approximation
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kg,
                zg,
                c_poly(kg,zg,η_opt),
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('z', fontsize=14)
plt.show()