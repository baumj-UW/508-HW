'''
Created on May 27, 2019

@author: Jackie Baum, Lane Smith
EE508 - Final Project
CT Markov Model

Sample component based on G. Anders "Optimal Maintenance Policies for Power Equipment"
Section 4.3.2 (p32)
'''
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt


## Q matrix based on HV Air-Blast Circuit Breakers (historical data)
## 13 States [D1, D2, D3, F, I1, I2, I3, M1, MM1, M2, MM2, M3, MM3]
## this matrix 
Q = np.array([[-0.01,0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0],\
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
              [3.044E-04, 0, 0, -3.044E-04, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
              [90.0, 0, 0, 0, -100.0, 0, 0, 5.0, 5.0, 0, 0, 0, 0],\
              [0, 0, 0, 0, 0, -100.0, 0, 0, 0, 90.0, 10.0, 0, 0],\
              [0, 0, 0, 0, 0, 0, -100.0, 0, 0, 0, 0, 10.0, 90.0],\
              [99.0, 1.0, 0, 0, 0, 0, 0, -100.0, 0, 0, 0, 0, 0],\
              [19.8, 0.2, 0, 0, 0, 0, 0, 0, -20.0, 0, 0, 0, 0],\
              [50.0, 44.0, 1.0, 0, 0, 0, 0, 0, 0, -95.0, 0, 0, 0],\
              [18.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, -20.0, 0, 0],\
              [0, 10.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, -100.0, 80.0],\
              [18.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -20.0]])


def SolveSS(Q):
    #Solve for stationary probabilities given Q matrix for CTMC
    M = np.insert(Q.T,Q.shape[0],1,axis=0)
    b = np.zeros((Q.shape[0]+1,1))
    b[-1,0] = 1
    px = lstsq(M,b,rcond=None)[0]
    return px[:,0] 


#Test Q Steady State
#solution: [4/7, 2/7, 1/7]
sample =  np.array([[-5,4,1],[10,-10,0],[0,4,-4]])
print(SolveSS(sample))

print('working?')