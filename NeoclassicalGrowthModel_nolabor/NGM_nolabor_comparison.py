# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:50:08 2023

@author: Eurhope
"""

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
os.chdir("C:\\Users\\Eurhope\\Dropbox\\PhD Bocconi\\2nd year courses\\Advanced Macroeconomics IV")
sys.path.append(f"{os.getcwd()}\\functions")
sys.path.append(f"{os.getcwd()}\\NGM_without_labor")


from functions_library import *
from NGM_nolabor_complete import NGM_nolabor_logpol




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
        


def chi_steady_state(params=Parameters()):
    
    ''' Get parameters '''
    alpha = params.alpha
    beta = params.beta
    # nu = params.nu
    delta = params.delta
    
    ''' Steady State Equations with Labor '''
    # k_ss = ((1/(alpha*(l_ss)**(1 - alpha)))*(1/beta - 1 + delta))**(1/(alpha-1))
    # c_ss = (k_ss**alpha)*(l_ss)**(1 - alpha) - delta*k_ss
    # chi = (1 - alpha)*(k_ss**alpha)*((l_ss**(-alpha - 1/nu)))*(1/c_ss)
    
    k_ss = (beta * alpha/(1-beta*(1-delta)))**(1/(1-alpha))
    c_ss = (k_ss**alpha) - delta*k_ss
    
    return k_ss, c_ss

""" Get Chi! """
k_ss,c_ss = chi_steady_state()




class Neoclassical_Growth_Model(Parameters):
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
        kmin = 0.5 * k_ss
        kmax = 1.5 * k_ss    
        
        self.kmin = kmin
        self.kmax = kmax

        
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
        
        # self.grid_k = collocation_nodes(kmin, kmax, n_k)
        # self.grid_z = np.exp(collocation_nodes(zmin, zmax, n_z))
        cheb_nodes_z = Chebyshev_Nodes(n_z)
        cheb_nodes_k = Chebyshev_Nodes(n_k)
        self.grid_k = np.exp(Change_Variable_Fromcheb(np.log(kmin), np.log(kmax), cheb_nodes_k))
        self.grid_z = np.exp(Change_Variable_Fromcheb(zmin, zmax, cheb_nodes_z))
        #c_star self.grid_k = np.linspace(kmin, kmax, n_k)
        # self.grid_z = np.exp(np.linspace(zmin, zmax, n_z))
        self.q_nodes, self.q_weights =  np.polynomial.hermite.hermgauss(n_q)  
        
        
        # """ Full alternating grid """
        # zk, kz = np.meshgrid(self.grid_z,self.grid_k)
        # self.grid_zk = np.array((zk.ravel()[::-1], kz.T.ravel()[::-1])).T

        """ Full alternating grid """
        _,zrep = np.meshgrid(self.grid_k,self.grid_z)
        zk, kz = np.meshgrid(self.grid_z,self.grid_k)
        self.grid_zk = np.array((zrep.ravel()[::-1], kz.T.ravel()[::-1])).T
        
        ''' Define another domanin for X_tilde (Herr page 305) '''
        
        lbounds = [zmin,np.log(kmin)]
        ubounds = [zmax,np.log(kmax)]
        # lbounds = [np.exp(zmin),kmin]
        # ubounds = [np.exp(zmax),kmax]
        self.lbounds = lbounds
        self.ubounds = ubounds

        
    # def approx_consumption_policy(self,gamma,x,y):

    #     consumption = c_poly(x,y,gamma)
        
    #     return consumption
        
        
        
        
    def approx_consumption_policy(self,gamma,x,y):

        p_z = self.p_z
        p_k = self.p_k
        
        # lbounds = np.log(self.lbounds)
        # ubounds = np.log(self.ubounds)

        lbounds = self.lbounds
        ubounds = self.ubounds

        #kron_zk = Tenser_Product_manual(lbounds,ubounds,x,y,p_z,p_k)
        kron_zk = Tenser_Product_manual(lbounds,ubounds,np.log(x),np.log(y),p_z,p_k)
        #consumption = c_poly(x,y,gamma)

        # kron_zk = Tenser_Product_manual(lbounds,ubounds,x,y,p_z,p_k)
        
        consumption = gamma @ kron_zk
        
        
        return np.exp(consumption)
        # return consumption

        
        
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
        
       
        print(np.sum((RHS - 1/cons_policy)**2))
        return np.sum((RHS - 1/cons_policy)**2)
    



# %% Solve the model using Projection Methods with Chebyshev

model = Neoclassical_Growth_Model()

gamma0 = np.zeros((1,model.n_k*model.n_z))

''' Find optimal gamma by minimizing the residual function '''
q = lambda x: model.Euler_error(x)

gamma_star = opt.minimize(q,gamma0,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xtol': 1e-10,'ftol': 1e-10}).x

# %% Solve the model using Projection Methods with logarithm approximation
model2 = NGM_nolabor_logpol()

gamma0 = np.zeros([1,6])[0]

''' Find optimal gamma by minimizing the residual function '''
q2 = lambda x: model2.Euler_error(x)

eta_star = opt.minimize(q2,gamma0,method='Nelder-Mead',options={'disp':True,'maxiter':100000,'xtol': 1e-10,'ftol': 1e-10}).x

# %% DEBUGGING TENSER PRODUCT WITH CHEB POLYNOMIALS


# x = model.grid_zk[:,0]
# y = np.reshape(model.grid_zk[:,1], (10,10)).T
# y = y.ravel()

# manual_kron = Tenser_Product_manual(model.lbounds,model.ubounds,x,y,10,10)

# kron_xy = Tenser_Product_new_points(model.grid_z[:,0],model.grid_k[:,0],10,10)

# cheb_nodes_x = Change_Variable_Tocheb(np.min(x), np.max(x), x)
# cheb_nodes_y = Change_Variable_Tocheb(np.min(y), np.max(y), y)

# ''' Multivariate case '''    
# p = 10
# x = cheb_nodes_x
# T = np.zeros([len(x),len(x)])
# T[0:p,:] = np.ones((p,len(x)))
# T[p:2*p,:] = np.reshape(np.repeat(x,p, 0).T,(p,len(x)))  

# b = np.reshape(np.repeat(x,p, 0).T,(p,len(x)))  

# for j in range(2,p):    
#     T[j*p:(j+1)*p,:] = 2*b*T[(j-1)*p:(j)*p,:] - T[(j-2)*p:(j-1)*p,:] 
    
  
    
# adj_cheb_nodes = np.reshape(cheb_nodes_x.T,(10,10)).T.ravel()
# T_x = Chebyshev_Polynomials_Recursion_mv(adj_cheb_nodes.T,10)
# T_y = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_y,10)

# T = Chebyshev_Polynomials_Recursion_x(cheb_nodes_x,10)
# # T_x_full = np.vstack([T_x]*10)
# # T_y_full = np.vstack([T_y]*10)

# # T_x_full = np.repeat(T_x,10, 0)
# # T_y_full = np.repeat(T_y,10, 0)

# T_x_full = np.repeat(T_x,10, 0)
# T_y_full = np.vstack([T_y]*10)

# # T_x_full = np.vstack([T_x]*10)
# # T_y_full = np.repeat(T_y,10, 0)
           
# manual_kron = T_x_full*T_y_full
# manual_kron_v2 = T*T_y_full

# %% small grid


# kron_xy = Tenser_Product_new_points(model.grid_z[:,0],model.grid_k[:,0],10,10)

# cheb_nodes_x_small = Change_Variable_Tocheb(np.min(model.grid_z[:,0]), np.max(model.grid_z[:,0]), model.grid_z[:,0])
# cheb_nodes_y_small = Change_Variable_Tocheb(np.min(model.grid_k[:,0]), np.max(model.grid_k[:,0]), model.grid_k[:,0])


# T_x_small = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_x_small,10)
# T_y_small = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_y_small,10)

# # I have to transpose T_x cause the first column as the first node, the second
# # column the second node and so on and so forth.
# # These below should be equivalent (just give a different format in Python)
# #tens_xy = np.tensordot(T_x.T,T_y.T, axes = 0)
# kron_xy_check = np.kron(T_x_small,T_y_small)

# %%
from matplotlib import cm # for 3d poltting

plt.close('all')
plt.style.use('seaborn') 

k_grid_fine = np.linspace(0.5 * k_ss,1.5 * k_ss,100)
z_grid_fine = np.exp(np.linspace(model.zmin,model.zmax,100))

# Generate meshgrid coordinates for 3d plot
zg, kg = np.meshgrid(z_grid_fine, k_grid_fine)
c_star = c_poly(zg,kg,eta_star)

""" Full alternating grid """
zk, kz = np.meshgrid(z_grid_fine,k_grid_fine)
picture_grid = np.array((zk.ravel()[::-1], kz.T.ravel()[::-1])).T
zk_2 = np.reshape(picture_grid[:,0],(100,100))
kz_2 = np.reshape(picture_grid[:,1],(100,100))

c_star_model = model.approx_consumption_policy(gamma_star, z_grid_fine, k_grid_fine)



cg = np.zeros(np.shape(zg))
for i in range(0,np.shape(zg)[1]):
    cg[:,i] = model.approx_consumption_policy(gamma_star, zk_2[:,i], kz_2[:,i])
       
# Plot policy function approximation
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(zg,
                kg,
                cg,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)

ax.plot_surface(zg,
                kg,
                c_star,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)

ax.set_xlabel('z', fontsize=14)
ax.set_ylabel('k', fontsize=14)

plt.show()



# k_grid_fine = np.linspace(0.5 * k_ss,1.5 * k_ss,100)
# z_grid_fine = np.exp(np.linspace(model.zmin,model.zmax,100))

# Generate meshgrid coordinates for 3d plot
zg, kg = np.meshgrid(z_grid_fine, k_grid_fine)
c_star = c_poly(zg,kg,gamma_star)

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
plt.show()