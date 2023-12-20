# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:02:11 2023

@author: peppecon
"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt


from scipy import optimize as opt
from sys import exit
# For Linux
# os.chdir("/home/nuagsire/Dropbox/PhD Bocconi/2nd year courses/Advanced Macroeconomics IV")
# sys.path.append(f"{os.getcwd()}/functions")

from numba import njit



# For Windows
os.chdir("C:\\Users\\Eurhope\\Dropbox\\PhD Bocconi\\2nd year courses\\Advanced Macroeconomics IV")
sys.path.append(f"{os.getcwd()}\\functions")

from functions_library import Tenser_Product_manual,Chebyshev_Nodes,Change_Variable_Tocheb


class Parameters:
    def __init__(self,alpha=0.33,
                 nu=2,
                 beta=0.99,
                 delta=0.025,
                 xi=1,
                 rho_z=0.95,
                 sigma_z=0.1):
        
        self.alpha = alpha
        self.nu = nu
        self.beta = beta
        self.delta = delta
        self.xi = xi
        self.rho_z = rho_z
        self.sigma_z = sigma_z
        
params = Parameters()
        


def chi_steady_state(l_ss, params=Parameters()):
    
    ''' Get parameters '''
    alpha = params.alpha
    beta = params.beta
    nu = params.nu
    delta = params.delta
    
    ''' Steady State Equations Modified '''
    k_ss = ((1/(alpha*(l_ss)**(1 - alpha)))*(1/beta - 1 + delta))**(1/(alpha-1))
    c_ss = (k_ss**alpha)*(l_ss)**(1 - alpha) - delta*k_ss
    chi = (1 - alpha)*(k_ss**alpha)*((l_ss**(-alpha - 1/nu)))*(1/c_ss)
    
    k_ss = (beta * alpha/(1-beta*(1-delta)))**(1/(1-alpha))
    c_ss = (k_ss**alpha) - delta*k_ss
    
    return k_ss, c_ss, chi

""" Get Chi! """
k_ss,c_ss,chi = chi_steady_state(0.33)

def c_poly(k,z,gamma):
    """
    Returns a polynomial approximation of consumption as a function of the state (k,z)
    """
    
    c_log = gamma[0] + gamma[1]*np.log(k) + gamma[2]*np.log(z) + gamma[3]*np.log(k)**2 + gamma[4]*np.log(z)**2 + gamma[5]*np.log(k)*np.log(z)
    
    return np.exp(c_log)


def grid(k_ss=k_ss,
                 n_z=10,
                 p_z=10,
                 n_k=10,
                 p_k=10,
                 n_q=5):
        
        # Explicitely call the super class to inherit parameterization
        sigma_z = params.sigma_z
        rho_z = params.rho_z
        
        # Setting up the capital grid
        kmin    =  0.5 * k_ss
        kmax   =  1.5 * k_ss    
        
        # Setting up the productivity grid (3 std)
        zmin = -3*np.sqrt(sigma_z**2/(1-rho_z**2))
        zmax = +3*np.sqrt(sigma_z**2/(1-rho_z**2))
        
        grid_k = np.linspace(kmin, kmax, n_k)
        grid_z = np.exp(np.linspace(zmin, zmax, n_z))
        q_nodes, q_weights =  np.polynomial.hermite.hermgauss(n_q)  
        
        zk, kz = np.meshgrid(grid_z,grid_k)
        grid_zk = np.array((zk.ravel(), kz.ravel())).T
        grid_kz = np.zeros((n_k*n_z,2))
        grid_kz[:,0] = grid_zk[:,1]
        grid_kz[:,1] = grid_zk[:,0]
        
        return grid_kz,q_nodes,q_weights,zmin,zmax,grid_k,grid_z
    

        
        
        
def output_policy(k,z):
        alpha = params.alpha
        output = z*(k**alpha)        
        return output
    
def next_period_capital(cons_policy,k,z):        
        delta = params.delta
        output = output_policy(k,z)        
        k_prime = output - cons_policy + (1-delta)*k                        
        return k_prime
        
grid_kz,q_nodes,q_weights,zmin,zmax,grid_k,grid_z = grid() 


""" Euler Error is vectorized! """
def Euler_error(gamma):
        
        grid_k_full = grid_kz[:,0]
        grid_z_full = grid_kz[:,1]
        alpha = params.alpha
        beta = params.beta
        sigma_z = params.sigma_z
        rho_z = params.rho_z
        delta = params.delta
        xi = params.xi
        
        ''' Compute Consumption policy given gamma '''
        cons_policy = c_poly(grid_k_full,grid_z_full,gamma)
        
        ''' Compute next period capital given consumption '''
        k_prime = next_period_capital(cons_policy,grid_k_full,grid_z_full) 

        RHS = 0    
        
        # vectorize this
        for node in range(len(q_nodes)):
            
            e_prime = np.sqrt(2)*sigma_z*q_nodes[node]        
            z_prime = np.exp(rho_z*np.log(grid_z_full) + xi*e_prime)           
                       
            cons_next = c_poly(k_prime,z_prime,gamma)
                       
            RHS += q_weights[node]*(beta*(cons_next**(-1))*(alpha*z_prime*(k_prime**(alpha-1)) + 1 - delta))
        
        RHS = RHS/np.sqrt(np.pi)
        
           
        return np.sum((RHS - cons_policy**(-1))**2)

def Euler_error_forloop(gamma):
        
        alpha = params.alpha
        beta = params.beta
        sigma_z = params.sigma_z
        rho_z = params.rho_z
        delta = params.delta
        xi = params.xi
        

        ssr = 0    
                      
        for i_k in range(len(grid_k)):  # Iterate over k and z grids
            
            for i_z in range(len(grid_z)):
                
                k       = grid_k[i_k]
                z       = grid_z[i_z]
                c       = c_poly(k,z,gamma)
                k_prime = next_period_capital(c,k,z)
                
                # Calculating the expectation over the GH nodes for every (k,z) weighted by the GH weights
                # We use the Gauss-Hermite formula with a change of variable
                
                E  = 0
                
                for i_q in range(len(q_nodes)):
                    
                    e_prime = np.sqrt(2) * sigma_z * q_nodes[i_q]         # The errors are normally distributed with mean 0 and std σ
                    z_prime = np.exp(rho_z * np.log(z) + xi*e_prime)
                    c_prime = c_poly(k_prime,z_prime,gamma)
                    
                    E += q_weights[i_q] * beta * c_prime**(-1) * (alpha * z_prime * k_prime**(alpha-1) + (1-delta))            
                    
                E = E / np.sqrt(np.pi)      
                ssr += (E - c**(-1))**2
         
           
        return ssr
    
    
gamma0 = np.zeros((1, 6))

''' Find optimal gamma by minimizing the residual function '''
q = lambda x: Euler_error(x)

# gamma0 = np.ones([1,len(model.grid_z)*len(model.grid_k)])*c_ss
gamma_star = opt.minimize(q,gamma0,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xtol': 1e-10,'ftol': 1e-10}).x

# %%
from matplotlib import cm # for 3d poltting

k_grid_fine = np.linspace(0.5 * k_ss,1.5 * k_ss,100)
z_grid_fine = np.exp(np.linspace(zmin,zmax,100))

# Generate meshgrid coordinates for 3d plot
kg, zg = np.meshgrid(k_grid_fine, z_grid_fine)
c_star = c_poly(kg,zg,gamma_star)

# Plot policy function approximation
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kg,
                zg,
                c_star,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('z', fontsize=14)
plt.show()