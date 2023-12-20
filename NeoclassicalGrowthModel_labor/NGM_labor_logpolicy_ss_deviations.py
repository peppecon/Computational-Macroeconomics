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
os.chdir("C:\\Users\\Eurhope\\Dropbox\\PhD Bocconi\\2nd year courses\\Advanced Macroeconomics IV")
sys.path.append(f"{os.getcwd()}\\functions")

from functions_library import *




""" Parameters """

class Parameters:
    def __init__(self,alpha=0.3,
                 nu=2,
                 beta=0.99,
                 delta=0.025,
                 xi=0.5,
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
    k_ss = ((1/(alpha*((l_ss)**(1 - alpha))))*(1/beta - 1 + delta))**(1/(alpha-1))
    c_ss = (k_ss**alpha)*(l_ss)**(1 - alpha) - delta*k_ss
    chi = (1 - alpha)*(k_ss**alpha)*((l_ss**(-alpha - 1/nu)))*(1/c_ss)
    
    return k_ss, c_ss, chi

""" Get Chi! """
k_ss,c_ss,chi = chi_steady_state(0.33)

params = Parameters()
alpha = params.alpha
beta = params.beta
nu = params.nu
delta = params.delta

theta = 0.357

phi = (1/alpha*(1/beta - 1 + delta))**(1/(1 -  alpha))
omega = phi**(1-alpha) - delta
psi = theta/(1-theta)*(1-alpha)*phi**(-alpha)

k_ss = psi/(omega + phi*psi)
l_ss = phi*k_ss
c_ss = omega*k_ss
y_ss = k_ss**alpha*l_ss**(1 - alpha)

#k_ss = 23.14

def c_poly(z,k,gamma):
    """
    Returns a polynomial approximation of consumption as a function of the state (z,k)
    """
        
    c_log = gamma[0] + gamma[1]*np.log(z) + gamma[2]*np.log(k) + gamma[3]*np.log(z)**2 + gamma[4]*np.log(k)**2 + gamma[5]*np.log(z)*np.log(k)

    
    # c_log = gamma[0] + gamma[1]*np.log(z) + gamma[2]*np.log(k) + gamma[3]*np.log(z)**2 + gamma[4]*np.log(k)**2 + gamma[5]*np.log(z)*np.log(k)\
    #         + gamma[6]*np.log(z)**3 + gamma[7]*np.log(k)**3 + gamma[8]*(np.log(z)**2*np.log(k)**2) + gamma[9]*np.log(z)**4 + gamma[10]*np.log(k)**4\
    #             + gamma[11]*(np.log(z)**3*np.log(k)**3)
    
    
    # return np.exp(c_log+np.log(c_ss))
    return np.exp(c_log)


class NGM_logpolicy(Parameters):
    def __init__(self,
                 chi=chi,
                 k_ss=k_ss,
                 n_z=13,
                 p_z=10,
                 n_k=13,
                 p_k=10,
                 n_q=5):
        
        # Explicitely call the super class to inherit parameterization
        Parameters.__init__(self)
        
        # Setting up the capital grid
        kmin    =  0.2 * k_ss
        kmax   =  1.8 * k_ss    
        
        self.kmin = kmin
        self.kmax = kmax
        
        self.chi = chi

        
        # Polynomial order
        self.p_z = p_z
        self.p_k = p_k
        
        # Grid details
        self.n_z = n_z
        self.n_k = n_k
        
        # Setting up the productivity grid (3 std)
        zmin = -3*np.sqrt(self.sigma_z**2/(1-self.rho_z**2))
        zmax = +3*np.sqrt(self.sigma_z**2/(1-self.rho_z**2))
        
        self.zmin = zmin
        self.zmax = zmax
        
        """ Define the collocation points """
        cheb_nodes_z = Chebyshev_Nodes(n_z)
        cheb_nodes_k = Chebyshev_Nodes(n_k)
        # self.grid_k = np.exp(Change_Variable_Fromcheb(np.log(kmin), np.log(kmax), cheb_nodes_k))
        self.grid_z = np.exp(Change_Variable_Fromcheb(zmin, zmax, cheb_nodes_z))
        self.grid_k = np.exp(Change_Variable_Fromcheb(np.log(kmin/k_ss), np.log(kmax/k_ss), cheb_nodes_k) + np.log(k_ss))
        
        self.q_nodes, self.q_weights =  np.polynomial.hermite.hermgauss(n_q)  
        
        
        zk, kz = np.meshgrid(self.grid_z,self.grid_k)
        # grid_zk = np.array((zk.ravel(), kz.ravel())).T
        grid_zk = np.array((zk.ravel()[::-1], kz.ravel()[::-1])).T

        # # grid_kz = np.zeros((n_k*n_z,2))
        # # grid_kz[:,0] = grid_zk[:,1]
        # # grid_kz[:,1] = grid_zk[:,0]
        # # self.grid_kz = grid_kz
        self.grid_zk = grid_zk
        
        # _,zrep = np.meshgrid(self.grid_k,self.grid_z)
        # zk, kz = np.meshgrid(self.grid_z,self.grid_k)
        # self.grid_zk = np.array((zrep.ravel()[::-1], kz.T.ravel()[::-1])).T

        
        
        
        
    def approx_consumption_policy(self,gamma,x,y):

        
        consumption = c_poly(x,y/k_ss,gamma)

        
        return consumption

    
    def labor_policy(self,cons_policy,z,k):
        
        chi = self.chi
        nu = self.nu
        alpha = self.alpha
        
        labor = (((1 - alpha)/(chi*cons_policy))*z*(k**alpha))**(nu/(1 + alpha*nu))
        
        return labor
        
        
    def output_policy(self,l,z,k):
        
        alpha = self.alpha       
       
        output = z*(k**alpha)*(l**(1-alpha))
        
        return output
    
    def next_period_capital(self,c,y,z,k):
        
        delta = self.delta
        
        k_prime = y - c + (1-delta)*k
                        
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
        n_z = self.n_z
        n_k = self.n_k
        
        ''' Compute Consumption policy given gamma '''
        cons_policy = self.approx_consumption_policy(gamma,grid_z_full,grid_k_full)
        
        ''' Compute Labor policy given gamma '''
        labor_policy = self.labor_policy(cons_policy, grid_z_full, grid_k_full)
        
        ''' Compute Output policy given gamma '''        
        output_policy = self.output_policy(labor_policy,grid_z_full,grid_k_full)

        
        ''' Compute next period capital given consumption '''
        k_prime = self.next_period_capital(cons_policy,output_policy,grid_z_full,grid_k_full) 

        RHS = 0    
        
        # vectorize this
        for node in range(len(q_nodes)):
            
            e_prime = np.sqrt(2)*sigma_z*q_nodes[node]        
            z_prime = np.exp(rho_z*np.log(grid_z_full) + xi*e_prime)           
                       
            cons_next = self.approx_consumption_policy(gamma,z_prime,k_prime)
            labor_next = self.labor_policy(cons_next, z_prime, k_prime)
                       
            RHS += q_weights[node]*(beta*(cons_policy/cons_next)*(alpha*z_prime*(k_prime**(alpha-1))*(labor_next**(1-alpha)) + 1 - delta))
            # RHS += q_weights[node]*(beta*(1/cons_next)*(alpha*z_prime*(k_prime**(alpha-1))*(labor_next**(1-alpha)) + 1 - delta))
        
        RHS = RHS/np.sqrt(np.pi)
        
            
        return np.sum((RHS - np.ones((n_z*n_k,1)).T)**2)
        # return np.sum((RHS - 1/cons_policy)**2)
    



# %% Solve the model using Projection Methods

model = NGM_logpolicy()

ncoefs = 6
gamma0 = np.zeros([1,ncoefs])[0]

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
l_star = model.labor_policy(c_star, zg, kg)

# # Plot policy function approximation
# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(zg,
#                 kg,
#                 c_star,
#                 rstride=2, cstride=2,
#                 cmap=cm.jet,
#                 alpha=0.5,
#                 linewidth=0.25)
# ax.set_xlabel('z', fontsize=14)
# ax.set_ylabel('k', fontsize=14)
# plt.show()

# Plot policy function approximation
subtitle_font=16

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(kg,
                zg,
                c_star,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('z', fontsize=14)
ax.set_title('Consumption Policy Function',fontsize=subtitle_font)


subtitle_font=16
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(kg,
                zg,
                l_star,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('z', fontsize=14)
ax.set_title('Labor Policy Function',fontsize=subtitle_font)

plt.show()
