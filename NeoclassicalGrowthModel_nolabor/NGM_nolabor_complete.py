#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:19:12 2023

@author: nuagsire
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:17:08 2023

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


# For Windows
os.chdir("C:\\Users\\Eurhope\\Dropbox\\PhD Bocconi\\2nd year courses\\Advanced Macroeconomics IV\\NGM_without_labor")
sys.path.append(f"{os.getcwd()}\\functions")

from functions_library import Tenser_Product_manual,Chebyshev_Nodes,Change_Variable_Tocheb




""" Parameters """

class Parameters:
    def __init__(self,alpha=0.30,
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
        


def chi_steady_state(l_ss, params=Parameters()):
    
    ''' Get parameters '''
    alpha = params.alpha
    beta = params.beta
    nu = params.nu
    delta = params.delta
    
    ''' Steady State Equations Modified '''
    # k_ss = ((1/(alpha*(l_ss)**(1 - alpha)))*(1/beta - 1 + delta))**(1/(alpha-1))
    # c_ss = (k_ss**alpha)*(l_ss)**(1 - alpha) - delta*k_ss
    # chi = (1 - alpha)*(k_ss**alpha)*((l_ss**(-alpha - 1/nu)))*(1/c_ss)
    
    k_ss = (beta * alpha/(1-beta*(1-delta)))**(1/(1-alpha))
    c_ss = (k_ss**alpha) - delta*k_ss
    
    return k_ss, c_ss

""" Get Chi! """
k_ss,c_ss = chi_steady_state(0.33)


def c_poly(z,k,gamma):
    """
    Returns a polynomial approximation of consumption as a function of the state (k,z)
    """
    
    c_log = gamma[0] + gamma[1]*np.log(z) + gamma[2]*np.log(k) + gamma[3]*np.log(z)**2 + gamma[4]*np.log(k)**2 + gamma[5]*np.log(z)*np.log(k)
    
    return np.exp(c_log)


class NGM_nolabor_logpol(Parameters):
    def __init__(self,
                 k_ss=k_ss,
                 n_z=10,
                 p_z=10,
                 n_k=10,
                 p_k=10,
                 n_q=5):
        
        # Explicitely call the super class to inherit parameterization
        Parameters.__init__(self)
        
        # Setting up the capital grid
        kmin    =  0.5 * k_ss
        kmax   =  1.5 * k_ss    
        
        self.kmin = kmin
        self.kmax = kmax

        
        # Polynomial order
        self.p_z = p_z
        self.p_k = p_k
        
        # Grid details
        self.n_z = n_z
        self.n_k = n_k
        
        # Setting up the productivity grid (3 std)
        zmin = -2*np.sqrt(self.sigma_z**2/(1-self.rho_z**2))
        zmax = +2*np.sqrt(self.sigma_z**2/(1-self.rho_z**2))
        
        self.zmin = zmin
        self.zmax = zmax
        
        self.grid_k = np.linspace(kmin, kmax, n_k)
        self.grid_z = np.exp(np.linspace(zmin, zmax, n_z))
        self.q_nodes, self.q_weights =  np.polynomial.hermite.hermgauss(n_q)  
        
        
        zk, kz = np.meshgrid(self.grid_z,self.grid_k)
        grid_zk = np.array((zk.ravel(), kz.ravel())).T
        # grid_kz = np.zeros((n_k*n_z,2))
        # grid_kz[:,0] = grid_zk[:,1]
        # grid_kz[:,1] = grid_zk[:,0]
        # self.grid_kz = grid_kz
        self.grid_zk = grid_zk

        
        
        
        
    def approx_consumption_policy(self,gamma,x,y):

        
        consumption = c_poly(x,y,gamma)

        
        return consumption

        
        
    def output_policy(self,z,k):
        
        alpha = self.alpha       
       
        output = z*(k**alpha)
        
        return output
    
    def next_period_capital(self,cons_policy,z,k):
        
        delta = self.delta
        output = self.output_policy(z,k)
        
        k_prime = output - cons_policy + (1-delta)*k
                        
        return k_prime
    
    
    def Euler_error(self,gamma):
        
        grid_z_full = self.grid_zk[:,0]
        grid_k_full = self.grid_zk[:,1]
        alpha = self.alpha
        beta = self.beta
        sigma_z = self.sigma_z
        q_nodes = self.q_nodes
        q_weights = self.q_weights
        rho_z = self.rho_z
        delta = self.delta
        xi = self.xi
        
        ''' Compute Consumption policy given gamma '''
        cons_policy = self.approx_consumption_policy(gamma,grid_z_full,grid_k_full)
        
        ''' Compute next period capital given consumption '''
        k_prime = self.next_period_capital(cons_policy,grid_z_full,grid_k_full) 

        RHS = 0    
        
        # vectorize this
        for node in range(len(q_nodes)):
            
            e_prime = np.sqrt(2)*sigma_z*q_nodes[node]        
            z_prime = np.exp(rho_z*np.log(grid_z_full) + xi*e_prime)           
                       
            cons_next = self.approx_consumption_policy(gamma,z_prime,k_prime)
                       
            RHS += q_weights[node]*(beta*(1/cons_next)*(alpha*z_prime*(k_prime**(alpha-1)) + 1 - delta))
        
        RHS = RHS/np.sqrt(np.pi)
        
            
        return np.sum((RHS - 1/cons_policy)**2)
    



# %% Solve the model using Projection Methods

model = NGM_nolabor_logpol()

gamma0 = np.zeros([1,6])[0]

''' Find optimal gamma by minimizing the residual function '''
q = lambda x: model.Euler_error(x)

eta_star = opt.minimize(q,gamma0,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xtol': 1e-10,'ftol': 1e-10}).x


# %%
from matplotlib import cm # for 3d poltting

k_grid_fine = np.linspace(0.5 * k_ss,1.5 * k_ss,100)
z_grid_fine = np.exp(np.linspace(model.zmin,model.zmax,100))

# Generate meshgrid coordinates for 3d plot
zg, kg = np.meshgrid(z_grid_fine,k_grid_fine)
c_star = c_poly(zg,kg,eta_star)

# Plot policy function approximation
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(zg,
                kg,
                c_star,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('z', fontsize=14)
ax.set_ylabel('k', fontsize=14)
plt.savefig('latex_problem_set/cpol_logpolicy.png')
plt.show()
