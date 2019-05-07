'''
Created on Apr 8, 2019

@author: baumj

EE 508 - HW1
Problem 4: Random Number generator
For defined discrete RV distribution, generate T iid variables
'''

import numpy as np
import matplotlib.pyplot as plt

# Define discrete RV
N = 5 # number of discrete values

A = [4,6,7,3,5] #given values for discrete RV
prob_A = 1/N*np.ones((N,1)) #uniform distrib. array of probabilities

def RandGen(t,prob,vals):
    '''
    t = number of iid draws to return, int
    prob = array of discrete RV probabilities
    vals = array of discrete RVs
    '''
    unif_draws = np.random.random(t)
    draws = np.zeros(unif_draws.shape)
    for n in range(t):
        for i in range(prob.size):
            if sum(prob[:i])<= unif_draws[n] < sum(prob[:i+1]):
                draws[n] = vals[i]
    return draws

test = RandGen(3,prob_A,A)
print(test)


prob_B = np.array([0.25,0.3,0.1,0.25,0.1])
B = [6,3,1,5,9]
testB = RandGen(1000,prob_B,B)


plt.hist(testB,bins=2*N,density=True)
plt.title('Histogram of Random Number Generator')
plt.xlabel('Discrete Outputs')
plt.ylabel('Sample Probability')
plt.show()

print("working?")