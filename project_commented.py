
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:21:22 2018
 
@author: twilmots, nicholasgoglio, johannabrahams
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import time
from py_vollib.black_scholes.implied_volatility import implied_volatility 
from scipy.stats.kde import gaussian_kde
rho = -0.7165
r = 0
S0 = 100
v_bar = 0.0354
v0 = v_bar 
l = 1.3253
eta = 0.3877
dt = 1/2500
N = 5
q = 0
K = 100
 
# Vectorizing the Implied Volatility function so it can operate on matricies
iv_function = np.vectorize(implied_volatility)
 
# Running Monte Carlo to simulate N paths that follow the Heston Stochastic Volatility Model
def monte_carlo_heston (N):
     
    var_matrix = np.zeros((N,2500)) # creating emtpy variance matrix with N rows and one column for each time step
    var_matrix[:,0] = v0 # intializing t = 0 to the initial variance
    stock_matrix = np.zeros((N,2500))
    stock_matrix[:,0] = np.log(S0)
 
    # Option prices matrices with strikes K = 90,95,100,105,110
    options_matrix_90 = np.zeros((N,2500))
    options_matrix_95 = np.zeros((N,2500))
    options_matrix_100 = np.zeros((N,2500))
    options_matrix_105 = np.zeros((N,2500))
    options_matrix_110 = np.zeros((N,2500))
 
    # Calculating Heston Model options prices at time = 0
    options_matrix_90[:,0] = heston_price(var_matrix[:,0],tau_m[:,0],np.exp(stock_matrix[:,0]),90)
    options_matrix_95[:,0] = heston_price(var_matrix[:,0],tau_m[:,0],np.exp(stock_matrix[:,0]),95)
    options_matrix_100[:,0] = heston_price(var_matrix[:,0],tau_m[:,0],np.exp(stock_matrix[:,0]),100)
    options_matrix_105[:,0] = heston_price(var_matrix[:,0],tau_m[:,0],np.exp(stock_matrix[:,0]),105)
    options_matrix_110[:,0] = heston_price(var_matrix[:,0],tau_m[:,0],np.exp(stock_matrix[:,0]),110)
     
     
    z = np.random.multivariate_normal([0,0],[[1,rho],[rho,1]], size = (N,2500)) # generating two RV for the two brownian motions
     
    # Iterating over all time steps and updating variance matrix, stock matrix and option price matrices
    for i in range (1,2500):
        var_matrix[:,i] = var_matrix[:,i-1] -l*(var_matrix[:,i-1]-v_bar)*dt+eta*np.sqrt(var_matrix[:,i-1])*np.sqrt(dt)*z[:,i,0] + (eta**2)/4*dt*(z[:,i,0]**2-1)
        stock_matrix[:,i] = stock_matrix[:,i-1] + (r-var_matrix[:,i]/2)*dt + np.sqrt(var_matrix[:,i]*dt)*z[:,i,1]
        options_matrix_90[:,i] = heston_price(var_matrix[:,i],tau_m[:,i],np.exp(stock_matrix[:,i]),90)
        options_matrix_95[:,i] = heston_price(var_matrix[:,i],tau_m[:,i],np.exp(stock_matrix[:,i]),95)
        options_matrix_100[:,i] = heston_price(var_matrix[:,i],tau_m[:,i],np.exp(stock_matrix[:,i]),100)
        options_matrix_105[:,i] = heston_price(var_matrix[:,i],tau_m[:,i],np.exp(stock_matrix[:,i]),105)
        options_matrix_110[:,i] = heston_price(var_matrix[:,i],tau_m[:,i],np.exp(stock_matrix[:,i]),110)
    stock_matrix = np.exp(stock_matrix)
    return stock_matrix, var_matrix, options_matrix_90,options_matrix_95,options_matrix_100,options_matrix_105,options_matrix_110
 
# Function for calculating Black-Scholes delta 
def Delta(S,sigma,tau):
    return ss.norm(0,1).cdf((np.log(S/K)+(sigma**2/2)*tau)/(sigma*np.sqrt(tau)))
 
# Profit and Loss function --> calculating profit and loss based on position of portfolio
def PnL (options,stocks,deltas):
    return options[:,1:]-options[:,:-1] - (stocks[:,1:]-stocks[:,:-1])*deltas[:,:-1]
 
# Function inside the intergral in Heston's formula
def heston_integrand(u,j,vt,tau,S,K):
    alpha = -(u**2)/2 + ( -u/2+j*u)*1j
    beta = l-j*eta*rho + (-rho*eta*u)*1j
    gamma = eta**2/2
    x = np.log(S/K)
    d = np.sqrt(beta**2-4*alpha*gamma)
    rplus = (beta + d)/(2*gamma)
    rminus = (beta - d)/(2*gamma)
    g = rminus/rplus
    D = rminus * (1-np.exp(-d*tau))/(1-g*np.exp(-d*tau))
    C = l*(rminus*tau - (2/eta**2)*np.log((1-g*np.exp(-d*tau))/(1-g)))
    return np.real(np.exp(C*v_bar+D*vt+u*x*1j)/(u*1j))
 
# Uses trapezoid integration to compute Pj for j = 0,1
def heston_P(j,vt,tau,S,K):
    u_range = np.array([np.arange(0.0001,100.0001,0.1),]*N)
    tau = np.array([tau,]*999).reshape(N,999)
    S = np.transpose(np.array([S,]*999))
    vt = np.transpose(np.array([vt,]*999))
    I =  (heston_integrand(u_range[:,1:],j,vt,tau,S,K) + heston_integrand(u_range[:,:999],j,vt,tau,S,K))*0.05
    I = I.sum(axis=1)
    return 1/2 + I/np.pi
 
# Computes prices from Heston formula, checks that Ps are [0,1] and returns zero if prices < 0
def heston_price(vt,tau,S,K):
    P0 = heston_P(0,vt,tau,S,K)
    P0 = P0*(P0>0)
    P0[np.where(P0>1)] = 1
    P1 = heston_P(1,vt,tau,S,K)
    P1 = P1*(P1>0)
    P1[np.where(P1>1)] = 1
    res_m = S*P1-K*P0
    res_m[np.where((res_m -S+K )< 0)] = S[np.where((res_m -S+K )< 0)]-K
    return res_m*(res_m>0) + 0
 
 
tau = np.array(np.arange(0,1,1/2500)) # time step matrix 
tau_m = 1 -  np.array([tau,]*N) 
del tau
start = time.time() # Timing entire monte carlo processes out of curiosity
stock_matrix,var_matrix, options_matrix_90,options_matrix_95,options_matrix,options_matrix_105,options_matrix_110 = monte_carlo_heston(N)
print("Time taken = {0:.5f}".format(time.time() - start))
 
# Calculating implied volatilty at each time step for each path
IV_matrix = iv_function(options_matrix,stock_matrix,K,tau_m,0,'c')
delta_m = Delta(stock_matrix,IV_matrix, tau_m ) #sigma_m is implied vols 
 
# Computes smaller matrices for the lower frequency hedging strategy
# Every variable name ending in _250 is the lower hedging frequency strategy
index = np.array(list(np.arange(-1,2499,10))) # Creating index array to pick out every 100th element of larger matrix
index[0] = index[0]+1
stock_matrix_250 = stock_matrix[:,index]
var_matrix_250 = var_matrix[:,index]
options_matrix_250 = options_matrix[:,index]
delta_m_250 = delta_m[:,index]
IV_matrix_250 = IV_matrix[:,index]
tau_m_250 = tau_m[:,index]
 
################################################## Question 1 ##################################################
 
# PnL for 1/2500 hedging strategy
pnl_A = PnL (options_matrix,stock_matrix,delta_m)
final_pnl_A = pnl_A.sum(axis=1) # summing up PnL across all time timesteps for each path
plt.hist(final_pnl_A) # plotting histogram of PnL to simulate density at time T
 
# PnL for 1/250 hedging strategy
pnl_A_250 = PnL(options_matrix_250,stock_matrix_250,delta_m_250)
final_pnl_A_250 = pnl_A_250.sum(axis=1)
plt.hist(final_pnl_A_250)
 
# Calculating returns for each time step and hedging strategy
returns_matrix = np.log(stock_matrix[:,1:])-np.log(stock_matrix[:,0:-1])
returns_matrix_250 = np.log(stock_matrix_250[:,1:])-np.log(stock_matrix_250[:,:-1])
 
var_vector = returns_matrix.var(axis=1)*2500
var_vector_250 = returns_matrix_250.var(axis=1)*250
 
# Plotting variance against PnL for each hedging strategy
plt.scatter(var_vector,final_pnl_A)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
plt.scatter(var_vector_250,final_pnl_A_250)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
 
################################################## Question 2 ##################################################
 
# Generating the realized return volatility matrices (RV - Realized Return Volatility)
rv_matrix = np.hstack((np.full((N,1),v0),np.sqrt((((stock_matrix[:,1:] - stock_matrix[:,:-1])**2)/(stock_matrix[:,:-1]**2))*2500)))
rv_matrix_250 = np.hstack(((np.full((N,1),v0)),np.sqrt((((stock_matrix_250[:,1:] - stock_matrix_250[:,:-1])**2)/(stock_matrix_250[:,:-1]**2))*250)))
 
# Calculating delta's for strategy B
delta_B = Delta(stock_matrix,rv_matrix,tau_m)
delta_B_250 = Delta(stock_matrix_250,rv_matrix_250,tau_m_250)
 
pnl_B = PnL(options_matrix,stock_matrix,delta_B)
final_pnl_B = pnl_B.sum(axis=1)
plt.hist(final_pnl_B)
 
pnl_B_250 = PnL(options_matrix_250,stock_matrix_250,delta_B_250)
final_pnl_B_250 = pnl_B_250.sum(axis=1)
plt.hist(final_pnl_B_250)
 
# Plotting continuous pdf and histogram to simulate time T PnL densities
pdf = gaussian_kde(final_pnl_B_250)
x = np.linspace(-10,10,100)
fig, ax1 = plt.subplots()
ax1.hist(final_pnl_B_250)
ax2 = ax1.twinx()
ax2.plot(x,pdf(x), color = 'r')
fig.tight_layout()
plt.show()
 
# Scatter plots for strategy B
plt.scatter(var_vector,final_pnl_B)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
plt.scatter(var_vector_250,final_pnl_B_250)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
# Computing time T densities of difference between strategy A and B PnL
difference_AB = final_pnl_A - final_pnl_B
difference_AB_250 = final_pnl_A_250 - final_pnl_B_250
 
plt.hist(difference_AB)
plt.hist(difference_AB_250)
 
# Calculating cumulative PnL for strategy A and B and for each hedging strategy
cum_pnl_A = np.cumsum(pnl_A,axis=1)
cum_pnl_A_250 = np.cumsum(pnl_A_250,axis=1)
cum_pnl_B = np.cumsum(pnl_B,axis=1)
cum_pnl_B_250 = np.cumsum(pnl_B_250,axis=1)
 
# Graphs comparing cumulative pnl and variance
j = N-1 # Path number (0<j<N-1)
fig, ax1 = plt.subplots()
ax1.plot(cum_pnl_A[j,:],color='black')
ax1.plot(cum_pnl_B[j,:])
ax2 = ax1.twinx()
#ax2.plot(var_matrix[j,:], color = 'red')
fig.tight_layout()
plt.show()
 
################################################## Question 3 ##################################################
 
# Generating a constant IV matrix so we can still use all our functions as we wrote them above
IV0 = np.full((N,2500),IV_matrix[0,0])
IV0_250 = np.full((N,250),IV_matrix[0,0])
 
# Generating deltas for strategy C
delta_C = Delta(stock_matrix,IV0,tau_m)
delta_C_250 = Delta(stock_matrix_250,IV0_250,tau_m_250)
 
# PnL for strategy C
pnl_C = PnL(options_matrix,stock_matrix,delta_C)
final_pnl_C = pnl_C.sum(axis=1)
plt.hist(final_pnl_C)
 
pnl_C_250 = PnL(options_matrix_250,stock_matrix_250,delta_C_250)
final_pnl_C_250 = pnl_C_250.sum(axis=1)
plt.hist(final_pnl_C_250)
 
 
# Scatter plots
plt.scatter(var_vector,final_pnl_C)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
plt.scatter(var_vector_250,final_pnl_C_250)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
# Differences between A,B and C
difference_AC = final_pnl_A - final_pnl_C
difference_AC_250 = final_pnl_A_250 - final_pnl_C_250
difference_BC = final_pnl_B - final_pnl_C
difference_BC_250 = final_pnl_B_250 - final_pnl_C_250
 
plt.hist(difference_AC)
plt.hist(difference_AC_250)
plt.hist(difference_BC)
plt.hist(difference_BC_250)
 
# Cumulative pnl paths for strategy C
 
cum_pnl_C = pnl_C.cumsum(axis=1)
cum_pnl_C_250 = pnl_C_250.cumsum(axis=1)
 
# Graphs for cumulative pnl and variance
j = N-1
fig, ax1 = plt.subplots()
ax1.plot(cum_pnl_A[j,:],color='black')
ax1.plot(cum_pnl_B[j,:],color='green')
ax1.plot(cum_pnl_C[j,:])
ax2 = ax1.twinx()
#ax2.plot(var_matrix[j,:], color = 'red')
fig.tight_layout()
plt.show()
 
################################################## Question 4 ##################################################
 
# Compute put prices from PCP
options_matrix_90 = options_matrix_90 - stock_matrix +np.full((N,2500),90)
options_matrix_95 = options_matrix_95 - stock_matrix +np.full((N,2500),95)
 
# Computes IVs for different strikes 
IV0_90 = np.full((N,2500),iv_function(options_matrix[0,0],S0,90,1,0,'p'))
IV0_95 = np.full((N,2500),iv_function(options_matrix[0,0],S0,95,1,0,'p'))
IV0_105 = np.full((N,2500),iv_function(options_matrix[0,0],S0,105,1,0,'c'))
IV0_110 = np.full((N,2500),iv_function(options_matrix[0,0],S0,110,1,0,'c'))
 
 
# 90 and 95 are puts so the delta is delta_call -1 
delta_D_90 = Delta(stock_matrix,IV0_90,tau_m) - 1
delta_D_95 = Delta(stock_matrix,IV0_95,tau_m) - 1
delta_D = Delta(stock_matrix,IV0,tau_m)
delta_D_105 = Delta(stock_matrix,IV0_105,tau_m)
delta_D_110 = Delta(stock_matrix,IV0_110,tau_m)
 
# TO DO: WE HAVE TO COME UP WITH A BETTER SCALING MECHANISM
pnl_D = ((1/90**2)*PnL(options_matrix_90,stock_matrix,delta_D_90) + (1/95**2)*PnL(options_matrix_95,stock_matrix,delta_D_95) + (1/100**2)*PnL(options_matrix
        ,stock_matrix,delta_D) + (1/105**2)*PnL(options_matrix_105,stock_matrix,delta_D_105)+ (1/110**2)*PnL(options_matrix_110,stock_matrix,delta_D_110))*(100**2)/5
 
final_pnl_D = pnl_D.sum(axis=1)
plt.hist(final_pnl_D)
 
# Doing all the same computations as above but for lower hedging frequency
options_matrix_90_250 = options_matrix_90 [:,index]
options_matrix_95_250 = options_matrix_95 [:,index]
options_matrix_105_250 = options_matrix_105 [:,index]
options_matrix_110_250 = options_matrix_110 [:,index]
 
IV0_90_250 = IV0_90[:,index]
IV0_95_250 = IV0_95[:,index]
IV0_105_250 = IV0_105[:,index]
IV0_110_250 = IV0_110[:,index]
 
delta_D_90_250 = delta_D_90[:,index]
delta_D_95_250 = delta_D_95[:,index]
delta_D_250 = delta_D[:,index]
delta_D_105_250 = delta_D_105[:,index]
delta_D_110_250 = delta_D_110[:,index]
 
pnl_D_250 = ((1/90**2)*PnL(options_matrix_90_250,stock_matrix_250,delta_D_90_250) + (1/95**2)*PnL(options_matrix_95_250,stock_matrix_250,delta_D_95_250) + (1/100**2)*PnL(options_matrix_250
        ,stock_matrix_250,delta_D_250) + (1/105**2)*PnL(options_matrix_105_250,stock_matrix_250,delta_D_105_250)+ (1/110**2)*PnL(options_matrix_110_250,stock_matrix_250,delta_D_110_250))*(100**2)/5
 
final_pnl_D_250 = pnl_D_250.sum(axis=1)
plt.hist(final_pnl_D_250)
 
#scatter plots
plt.scatter(var_vector,final_pnl_D)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
plt.scatter(var_vector_250,final_pnl_D_250)
plt.axvline(x=0.0354,color="r")
plt.axhline(y=0.0,color="black")
 
# Generatings PnL differences bewteen D and all other trading strategies
difference_DA = -final_pnl_A + final_pnl_D
difference_DA_250 = -final_pnl_A_250 + final_pnl_D_250
difference_DB = -final_pnl_B + final_pnl_D
difference_DB_250 = -final_pnl_B_250 + final_pnl_D_250
difference_DC = final_pnl_D - final_pnl_C
difference_DC_250 = final_pnl_D_250 - final_pnl_C_250
 
# Plotting time T difference densities
plt.hist(difference_DA)
plt.hist(difference_DA_250)
plt.hist(difference_DB)
plt.hist(difference_DB_250)
plt.hist(difference_DC)
plt.hist(difference_DC_250)
 
# Calculating cumulative sum of PnL for strategy D
cum_pnl_D = pnl_D.cumsum(axis=1)
cum_pnl_D_250 = pnl_D_250.cumsum(axis=1)
 
# Plotting cumulative PnL for each strategy for a specific stock price path j
j = N-1
fig, ax1 = plt.subplots()
ax1.plot(cum_pnl_A[j,:],color='black')
ax1.plot(cum_pnl_B[j,:],color='b')
ax1.plot(cum_pnl_C[j,:],color='c')
ax1.plot(cum_pnl_D[j,:],color='orange')
ax2 = ax1.twinx()
#ax2.plot(var_matrix[j,:], color = 'red')
fig.tight_layout()
plt.show()
