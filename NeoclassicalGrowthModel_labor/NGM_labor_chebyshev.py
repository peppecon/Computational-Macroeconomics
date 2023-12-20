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
# # For Linux
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
        
def get_z_bounds(n_z=10000,params=Parameters()):
    
    rho_z = params.rho_z
    sigma_z = params.sigma_z
    xi = params.sigma_z
    
    z_vec = np.zeros((n_z,1))
    
    for i in range(n_z-1):
        z_vec[i+1] = rho_z*z_vec[i] + sigma_z*np.random.normal(0,1)
        
    zmin = np.min(z_vec)
    zmax = np.max(z_vec)
    
    return zmin,zmax       


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
    
    return k_ss, c_ss, chi

""" Get Chi! """
k_ss,c_ss,chi = chi_steady_state(1/3)



class Neoclassical_Growth_Model(Parameters):
    def __init__(self,
                 chi,
                 k_ss=k_ss,
                 n_z=10,
                 p_z=10,
                 n_k=10,
                 p_k=10,
                 n_q=5):
        
        # Explicitely call the super class to inherit parameterization
        Parameters.__init__(self)
        
        # Setting up the capital grid
        kmin = 0.5*k_ss
        kmax = 1.5*k_ss
        self.kmin = kmin
        self.kmax = kmax
        
        # Polynomial order
        self.p_z = p_z
        self.p_k = p_k
        
        # Grid details
        self.n_z = n_z
        self.n_k = n_k
        
        # Setting up the productivity grid (3 std)
        # zmin = -3*np.sqrt(self.sigma_z**2/(1-self.rho_z**2))
        # zmax = 3*np.sqrt(self.sigma_z**2/(1-self.rho_z**2))
        zmin ,zmax = get_z_bounds()
        self.zmin = zmin
        self.zmax = zmax
        self.chi = chi
        # #self.grid_z = np.exp(np.linspace(zmin, zmax, n_z))
        # self.grid_k = np.linspace(kmin, kmax, n_k)
        # self.grid_z = np.linspace(zmin, zmax, n_z)
        # self.q_nodes, self.q_weights =  np.polynomial.hermite.hermgauss(n_q)  
               
        """ Define the collocation points """
        cheb_nodes_z = Chebyshev_Nodes(n_z)
        cheb_nodes_k = Chebyshev_Nodes(n_k)
        # self.grid_k = np.exp(Change_Variable_Fromcheb(np.log(kmin), np.log(kmax), cheb_nodes_k))
        self.grid_z = Change_Variable_Fromcheb(zmin, zmax, cheb_nodes_z)
        self.grid_k = np.exp(Change_Variable_Fromcheb(np.log(kmin/k_ss), np.log(kmax/k_ss), cheb_nodes_k) + np.log(k_ss))
        # self.grid_k = Change_Variable_Fromcheb(kmin, kmax, cheb_nodes_k)
        # self.grid_k = (Change_Variable_Fromcheb((kmin-k_ss)/k_ss, np.log(kmax/k_ss), cheb_nodes_k) + 1)*k_ss
        self.q_nodes, self.q_weights =  np.polynomial.hermite.hermgauss(n_q)  
        

        """ Grid with productivity chuncks"""
        _,zrep = np.meshgrid(self.grid_k,self.grid_z)
        zk, kz = np.meshgrid(self.grid_z,self.grid_k)
        self.grid_zk = np.array((zrep.ravel()[::-1], kz.T.ravel()[::-1])).T
        # self.grid_zk = np.array((zrep.ravel(), kz.T.ravel())).T
        
        # """ Normal Grid"""
        # zk, kz = np.meshgrid(self.grid_z,self.grid_k)
        # self.grid_zk = np.array((zk.ravel()[::-1], kz.ravel()[::-1])).T
        # self.grid_zk = np.array((zk.ravel(), kz.ravel())).T
        
        ''' Define another domanin for X_tilde (Herr page 305) '''
        
        # lbounds = [zmin-0.5,np.log(kmin/k_ss)-0.5]
        # ubounds = [zmax+0.5,np.log(kmax/k_ss)+0.5]
        lbounds = [zmin-0.1,np.log(kmin/k_ss-0.1)]
        ubounds = [zmax+0.1,np.log(kmax/k_ss+0.1)]
        # lbounds = [zmin,np.log(kmin/k_ss)]
        # ubounds = [zmax,np.log(kmax/k_ss)]
        # lbounds = [np.min(self.grid_z),np.log(np.min(self.grid_k)/k_ss)]
        # ubounds = [np.max(self.grid_z),np.log(np.max(self.grid_k)/k_ss)]
        # lbounds = [np.exp(zmin-0.3),kmin-1]
        # ubounds = [np.exp(zmax+0.3),kmax+1]
        # lbounds = [zmin,kmin]
        # ubounds = [zmax,kmax]
        self.lbounds = lbounds
        self.ubounds = ubounds
        
        
        
    def approx_consumption_policy(self,gamma,x,y):
        
        p_z = self.p_z
        p_k = self.p_k
        lbounds = self.lbounds
        ubounds = self.ubounds

        kron_zk = Tenser_Product_manual(lbounds,ubounds,x,np.log(y/k_ss),p_z,p_k)
        
        consumption = gamma.T @ kron_zk
        
        return np.exp(consumption + np.log(c_ss))
    
        
    def labor_policy(self,cons_policy,z,k):
        
        chi = self.chi
        nu = self.nu
        alpha = self.alpha
        
        labor = ((1/(chi*cons_policy))*(1 - alpha)*np.exp(z)*(k**alpha))**(nu/(1 + alpha*nu))
        
        return labor
        
        
    def output_policy(self,cons_policy,z,k):
        
        alpha = self.alpha
        labor_policy = self.labor_policy(cons_policy,z,k)
        
        labor = labor_policy
        
        output = np.exp(z)*(k**alpha)*(labor**(1-alpha))
        
        return output
    
    def next_period_capital(self,cons_policy,z,k):
        
        # grid_k_full = self.grid_zk[:,1]
        delta = self.delta
        grid_k = self.grid_k
        output = self.output_policy(cons_policy,z,k)
        
        k_prime = output - cons_policy + (1-delta)*k
                
        # for i in range(len(k_prime)):
        #     if k_prime[i] > np.max(grid_k):
        #         k_prime[i] = np.max(grid_k)
        #     elif k_prime[i] < np.min(grid_k):
        #         k_prime[i] = np.min(grid_k)
        #     else:
        #         continue
            
                                   
        return k_prime
    
    
    def Euler_error(self,gamma):
        
        grid_z_full = self.grid_zk[:,0]
        grid_k_full = self.grid_zk[:,1]
        n_z = self.n_z
        n_k = self.n_k
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
        
        output = self.output_policy(cons_policy,grid_z_full,grid_k_full)
        
        for i in range(len(cons_policy)):
            if cons_policy[i] > output[i]:
                cons_policy[i] = output[i]
            else:
                continue
        
        ''' Compute next period capital given consumption '''
        k_prime = self.next_period_capital(cons_policy,grid_z_full,grid_k_full) 
        
        

        RHS = 0    
        
        # vectorize this
        for node in range(len(q_nodes)):
            
            e_prime = np.sqrt(2)*sigma_z*q_nodes[node]            
            z_prime = rho_z*grid_z_full + e_prime
                        
            # for i in range(len(z_prime)):
            #     if z_prime[i] < np.max(grid_z_full):
            #         print("Productivity exits the grid!")
                           
            cons_next = self.approx_consumption_policy(gamma,z_prime,k_prime)
            
            output_next = self.output_policy(cons_next,z_prime,k_prime)

            
            for i in range(len(cons_policy)):
                if cons_next[i] > output_next[i]:
                    cons_next[i] = output_next[i]
                else:
                    continue
                                   
            labor_next = self.labor_policy(cons_next,z_prime,k_prime)
            
            RHS += q_weights[node]*(beta*(cons_policy/cons_next)*(alpha*np.exp(z_prime)*(k_prime**(alpha-1))*(labor_next**(1-alpha)) + 1 - delta))
            # RHS += q_weights[node]*(beta*(1/cons_next)*(alpha*np.exp(z_prime)*(k_prime**(alpha-1))*(labor_next**(1-alpha)) + 1 - delta))
        
        RHS = RHS/np.sqrt(np.pi)
        
        EE_residual = RHS - np.ones((n_z*n_k,1)).T
        # EE_residual = RHS - 1/cons_policy
        
        print("Residual = ",np.sum(EE_residual**2))
                    
        return np.sum(EE_residual**2)
    



# %% Solve the model using Projection Methods

model = Neoclassical_Growth_Model(chi)

grid_z_full = model.grid_zk[:,0]
grid_k_full = model.grid_zk[:,1]
n_z = model.n_z
n_k = model.n_k

''' Set up initial conditions '''
m = n_z*n_k + n_z*n_k

gamma0 = np.zeros([1,n_z*n_k])[0]

''' Find optimal gamma by minimizing the residual function '''
q = lambda x: model.Euler_error(x)

gamma_star = opt.minimize(q,gamma0,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xatol': 1e-5,'fatol': 1e-5}).x



# %%
''' Get Optimal Consumption '''

grid_z_full = model.grid_zk[:,0]
grid_k_full = model.grid_zk[:,1]
grid_z = model.grid_z
grid_k = model.grid_k
n_z = model.n_z
n_k = model.n_k


""" Plot the Consumption Policy Function """

plt.close('all')

''' Prepare the data from the approximated function '''
z_approx, k_approx = np.meshgrid(grid_z, grid_k)
c_star = model.approx_consumption_policy(gamma_star, grid_z_full, grid_k_full)
l_star = model.labor_policy(c_star,grid_z_full,grid_k_full)
check = np.array([grid_z_full,grid_k_full,c_star]).T
c_star = np.reshape(c_star, (n_k,n_z)).T
l_star = np.reshape(l_star, (n_k,n_z)).T



subtitle_font=16
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.plot_surface(z_approx, k_approx, c_star, edgecolor='blue', lw=0.8, alpha=0)
ax.set_title('Consumption Policy Function',fontsize=subtitle_font)

# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('Labor Policy Function',fontsize=subtitle_font)


ax.plot_surface(z_approx, k_approx, l_star, edgecolor='red', lw=0.8, alpha=0)
# plt.savefig('latex/clpol_fun_projection3d.png')
plt.show()




# %%
# from matplotlib import cm # for 3d poltting

# k_grid_fine = np.linspace(model.kmin,model.kmax,100)
# z_grid_fine = np.exp(np.linspace(model.zmin,model.zmax,100))

# # Generate meshgrid coordinates for 3d plot
# zg, kg = np.meshgrid(z_grid_fine,k_grid_fine)

# cg = np.zeros(np.shape(zg))
# lg = np.zeros(np.shape(zg))
# for i in range(0,np.shape(zg)[1]):
#     cg[:,i] = model.approx_consumption_policy(gamma_star, zg[:,i], kg[:,i])
#     lg[:,i] = model.labor_policy(cg[:,i],zg[:,i],kg[:,i])
    

# # Plot policy function approximation
# subtitle_font=16

# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_surface(kg,
#                 zg,
#                 cg,
#                 rstride=2, cstride=2,
#                 cmap=cm.jet,
#                 alpha=0.5,
#                 linewidth=0.25)
# ax.set_xlabel('k', fontsize=14)
# ax.set_ylabel('z', fontsize=14)
# ax.set_title('Consumption Policy Function',fontsize=subtitle_font)


# ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.plot_surface(kg,
#                 zg,
#                 lg,
#                 rstride=2, cstride=2,
#                 cmap=cm.jet,
#                 alpha=0.5,
#                 linewidth=0.25)
# ax.set_xlabel('k', fontsize=14)
# ax.set_ylabel('z', fontsize=14)
# ax.set_title('Labor Policy Function',fontsize=subtitle_font)

# plt.show()


# %%
n_sim = 100
dev_k = 4
k_grid_fine = np.linspace(model.kmin,model.kmax,n_sim)
c_star = model.approx_consumption_policy(gamma_star, np.zeros((n_sim)), k_grid_fine)
l_star = model.labor_policy(c_star,np.zeros((n_sim)),k_grid_fine)
plt.figure(figsize=(18, 14))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('K',fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.ylabel('C',fontsize=18)
plt.plot(k_grid_fine,c_star)
# plt.savefig('latex/cpol_fun_projection1.png')
plt.show()

n_sim = 100
z_grid_fine = np.linspace(model.zmin,model.zmax,100)
c_star = model.approx_consumption_policy(gamma_star, z_grid_fine, np.vstack([k_ss]*n_sim).ravel())
l_star = model.labor_policy(c_star,np.zeros((n_sim)),k_grid_fine)
plt.figure(figsize=(18, 14))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('z',fontsize=18)
plt.ylabel('C',fontsize=18)
plt.plot(z_grid_fine,c_star)
# plt.savefig('latex/cpol_fun_projection2.png')
plt.show()