#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:17:59 2020

@author: phuntrakaool
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

size = [1000, 5000, 10000, 20000]
color = ['r','y','b','g','m']

def p_e(x):
    if x < 0:
        return 0
    else:
        return (0.5*np.exp(-0.5*x)) * (1/math.factorial(5)*(x**4)*np.exp(-x))

y = 0
for n in size:
    x = np.arange(0.0,n*1.0)
    x[0] = 3.0
    
    plt.style.use('seaborn')
    
    #Metropolis-Hastings
    for i in range(1,n):
        currentx = x[i-1]
        proposedx = currentx + np.random.normal(0,1,1) #mean,std,size
        A = p_e(proposedx)/p_e(currentx)
        if np.random.rand() <= A:
            x[i] = proposedx
        else:
            x[i] = currentx
            
    #
    #plt.scatter(np.arange(0,n),x)
    #plt.show()
        
    sns.distplot(x, label = 'size '+str(n), color=color[y], hist = False)
    y+=1


plt.xticks(range(15))
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

#
#def mcmc(niter, startval, proposalsd):
#    x = np.arange(0.0,niter*1.0)
#    x[0] = startval
#    for i in range(1,niter):
#        currentx = x[i-1]
#        proposedx = np.random.normal(currentx,proposalsd,1)
#        A = p_e(proposedx)/p_e(currentx)
#        if np.random.rand() < A:
#            x[i] = proposedx
#        else:
#            x[i] = currentx 
#            
#    plt.scatter(np.arange(0,niter),x)
#    plt.show()
#    
#    plt.hist(x)
#    plt.xticks(range(5))
#    plt.show()
#    return x 
#
#z1 = mcmc(1000, 3, 1)
#z2 = mcmc(1000, 3, 1)
#z3 = mcmc(1000, 3, 1)
#
#plt.plot(z1,"r")
#plt.plot(z2,"g")
#plt.plot(z3,"b")
#plt.show()
#
#
#
#def prior(p):
#    if (p<0) | (p>1):
#        return 0
#    else:
#        return 1
#    
#def likelihood(p, nAA, nAa, naa):
#    return p**(2*nAA) * (2*p*(1-p))**nAa * (1-p)**(2*naa)
#
#
#
#def mcmc_sampler(nAA, nAa, naa, niter, startval, proposalsd):
#    x = np.arange(0.0,niter*1.0)
#    x[0] = startval
#    for i in range(1,niter):
#        currentx = x[i-1]
#        proposedx = currentx + np.random.normal(0, proposalsd,1)
#        A_num = prior(proposedx)*likelihood(proposedx,nAA,nAa,naa)
#        A_den = prior(currentx)*likelihood(currentx,nAA,nAa,naa)
#        A = A_num/A_den
#        if np.random.rand() < A:
#            x[i] = proposedx
#        else:
#            x[i] = currentx 
#            
#    plt.scatter(np.arange(0,niter),x)
#    plt.show()
#    
#    plt.hist(x)
#    plt.show()
#    return x 
#
#mcmc_sampler(50,21,29,10000,0.5,0.01)