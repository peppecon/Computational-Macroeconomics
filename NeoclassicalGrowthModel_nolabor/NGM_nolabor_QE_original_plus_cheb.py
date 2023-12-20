#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:00:23 2023

@author: nuagsire
"""

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

class Parameters:
    def __init__(self,alpha=0.3,
                 nu=2,
                 beta=0.99,
                 delta=0.025,
                 xi=1,
                 rho_z=0.95,
                 sigma_z=0.007):
        
        self.alpha = alpha
        self.nu = nu
        self.beta = beta
        self.delta = delta
        self.xi = xi
        self.rho_z = rho_z
        self.sigma_z = sigma_z

def chi_steady_state(params=Parameters()):
    
    ''' Get parameters '''
    alpha = params.alpha
    beta = params.beta
    delta = params.delta
    
    ''' Steady State Equations with Labor '''   
    k_ss = (beta * alpha/(1-beta*(1-delta)))**(1/(1-alpha))
    c_ss = (k_ss**alpha) - delta*k_ss
    
    return k_ss, c_ss

""" Get Chi! """
k_ss,c_ss = chi_steady_state()

@njit
def Change_Variable_Tocheb(bmin, bmax, b):    
    x = 2*b/(bmax - bmin) - (bmin+bmax)/(bmax-bmin)
    return x


@njit
def Change_Variable_Fromcheb(bmin, bmax, x):             
    # b_rev = (x + 1)*(bmax - bmin)/2 + bmin    
    b_rev = (bmin + bmax)/2 + ((bmax - bmin)/2)*x
    return b_rev

@njit
def Chebyshev_Nodes(n):    
    cheb_nodes = np.zeros((n,1))    
    for k in range(0,n):
        ''' We have to adjust the Chebyshev nodes formula because Python starts
            counting from 0, hence we need k+1, but the indexing is still k'''
        cheb_nodes[k] = np.cos(((2*(k+1)-1)/(2*n))*np.pi)    
    return cheb_nodes   

@njit
def Chebyshev_Polynomials_Recursion_uv(x,p):
    ''' Univariate case '''    
    T = np.zeros((p,1)).ravel()
    T[0] = 1
    T[1] = x
    for j in range(1,p-1):
        T[j+1] = 2*x*T[j] - T[j-1]
    return T

@njit
def c_cheb(k,z,gamma,k_low,k_high,z_low,z_high,p_k,p_z):
    
    k_cheb = Change_Variable_Tocheb(np.log(k_low/k_ss),np.log(k_high/k_ss),np.log(k/k_ss))
    # k_cheb = Change_Variable_Tocheb(np.log(k_low),np.log(k_high),np.log(k))
    # k_cheb = Change_Variable_Tocheb(k_low,k_high,k)      
    z_cheb = Change_Variable_Tocheb(z_low,z_high,np.log(z))
    # z_cheb = Change_Variable_Tocheb(np.exp(z_low),np.exp(z_high),z)
    T_k = Chebyshev_Polynomials_Recursion_uv(k_cheb,p_k)
    T_z = Chebyshev_Polynomials_Recursion_uv(z_cheb,p_z)
    kron_kz = np.kron(T_k,T_z)	
    c = gamma @ kron_kz
	
    return np.exp(c + np.log(c_ss))
    # return np.exp(c)
    # return c
    
@njit
def c_cheb_v2(k,z,gamma,k_low,k_high,z_low,z_high,p_k,p_z):
    
    k_cheb = 2*np.log(k/k_ss)/(np.log(k_high/k_ss) - np.log(k_low/k_ss)) - (np.log(k_low/k_ss)+np.log(k_high/k_ss))/(np.log(k_high/k_ss)-np.log(k_low/k_ss))   
    z_cheb = 2*np.log(z)/(z_high - z_low) - (z_low+z_high)/(z_high-z_low)
    
    T_k = np.zeros((p_k,1)).ravel()
    T_k[0] = 1
    T_k[1] = k_cheb
    for j in range(1,p_k-1):
        T_k[j+1] = 2*k_cheb*T_k[j] - T_k[j-1]
        
        
    T_z = np.zeros((p_z,1)).ravel()
    T_z[0] = 1
    T_z[1] = z_cheb
    for j in range(1,p_z-1):
        T_z[j+1] = 2*z_cheb*T_z[j] - T_z[j-1]
        
    
    kron_kz = np.kron(T_k,T_z)	
    c = gamma @ kron_kz
	
    return np.exp(c + np.log(c_ss))



def c_cheb_vec(k_vec,z_vec,gamma,k_low,k_high,z_low,z_high,p_k,p_z):
    
    c = np.zeros((np.shape(k_vec)))
    
    for i in range(0,np.shape(k_vec)[0]):
        for j in range(0,np.shape(z_vec)[1]):
            k = k_vec[0,i]
            z = z_vec[j,0]
            k_cheb = Change_Variable_Tocheb(np.log(k_low/k_ss),np.log(k_high/k_ss),np.log(k/k_ss))
            # k_cheb = Change_Variable_Tocheb(np.log(k_low),np.log(k_high),np.log(k))
            # k_cheb = Change_Variable_Tocheb(k_low,k_high,k)      
            # z_cheb = Change_Variable_Tocheb(np.exp(z_low),np.exp(z_high),z)
            z_cheb = Change_Variable_Tocheb(z_low,z_high,np.log(z))
            T_k = Chebyshev_Polynomials_Recursion_uv(k_cheb,p_k)
            T_z = Chebyshev_Polynomials_Recursion_uv(z_cheb,p_z)
            kron_kz = np.kron(T_k,T_z)	
            c[i,j] = gamma @ kron_kz
	
    return np.exp(c + np.log(c_ss))
    # return np.exp(c)
    # return c


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
    α = 0.3
    δ = 0.025
    γ = 1
    ρ = 0.95
    σ = 0.007
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
            
    print(ssr)
    return ssr

@njit
def euler_err_cheb(gamma,quad):
    """
    Returns the sum of squared Euler errors at all grid points
    
    """
    
    q_nodes, q_weights = quad
    β = 0.99
    α = 0.3
    δ = 0.025
    γ = 1
    ρ = 0.95
    σ = 0.007
    ssr      =  0  # Initialize the sum of squared errors
    
    for i_k in range(len(k_grid)):  # Iterate over k and z grids
        
        for i_z in range(len(z_grid)):
            
            k       = k_grid[i_k]
            z       = z_grid[i_z]
            c       = c_cheb_v2(k,z,gamma,k_low,k_high,z_low,z_high,p_k,p_z)
            
            # y = z*k**α
            
            # if c > y:
            #     c = y
            
            k_prime = z * k**α + (1-δ) * k - c;
            
            # if k_prime > k_high:
            #     k_prime = k_high
                        
            # Calculating the expectation over the GH nodes for every (k,z) weighted by the GH weights
            # We use the Gauss-Hermite formula with a change of variable
            
            E  = 0
            
            for i_q in range(len(q_nodes)):
                
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]         # The errors are normally distributed with mean 0 and std σ
                z_prime = np.exp(ρ * np.log(z) + e_prime)
                c_prime = c_cheb_v2(k_prime,z_prime,gamma,k_low,k_high,z_low,z_high,p_k,p_z)
                # y_prime = z_prime*k_prime**α
                
                # if c_prime > y_prime:
                #     c_prime = y_prime
                
                E += q_weights[i_q] * β * c_prime**(-γ) * (α * z_prime * k_prime**(α-1) + (1-δ))            
                
            E = E / np.sqrt(np.pi)      
            ssr += (E - c**(-γ))**2
            
    print(ssr)
    return ssr

β = 0.99
α = 0.3
δ = 0.025
γ = 1
ρ = 0.95
σ = 0.007
p_k = 10
p_z = 10

# Calculate the steady state level of capital
k_ss = (β * α/(1-β*(1-δ)))**(1/(1-α))

# Setting up the capital grid (with collocation points)
k_low    =  0.5 * k_ss
k_high   =  1.5 * k_ss
n_k =  10
# k_grid = np.linspace(k_low,k_high,n_k)
cheb_nodes_k = Chebyshev_Nodes(n_k)
k_grid = np.exp(Change_Variable_Fromcheb(np.log(k_low/k_ss), np.log(k_high/k_ss), cheb_nodes_k) + np.log(k_ss)).ravel()
# k_grid = np.exp(Change_Variable_Fromcheb(np.log(k_low), np.log(k_high), cheb_nodes_k)).ravel()
# k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k).ravel()

# Setting up the productivity grid (3 std) (with collocation points)
z_low    = -3 * np.sqrt(σ**2/(1-ρ**2))
z_high   =  3 * np.sqrt(σ**2/(1-ρ**2))
n_z =  10 
# z_grid  = np.exp(np.linspace(z_low,z_high,n_z))
cheb_nodes_z = Chebyshev_Nodes(n_z)
z_grid = np.exp(Change_Variable_Fromcheb(z_low, z_high, cheb_nodes_z)).ravel()

# Set initial values for the coefficients
η_init = np.zeros((1, 6))
gamma_0 = np.zeros((1,p_k*p_z))

# Find solution by minimizing the errors on the grid
η_opt = opt.minimize(euler_err,η_init, args = (gh_quad,) ,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xatol': 1e-10,'fatol': 1e-10}).x
print(η_opt)

# Find solution by minimizing the errors on the grid (Chebyshev)
gamma_star = opt.minimize(euler_err_cheb,gamma_0, args = (gh_quad,) ,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xatol': 1e-10,'fatol': 1e-10}).x
print(gamma_star)



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

c_cheb_approx = c_cheb_vec(kg,zg,gamma_star,k_low,k_high,z_low,z_high,p_k,p_z)

# Plot policy function approximation with Chebyshev
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kg,
                zg,
                c_cheb_approx,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('z', fontsize=14)

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