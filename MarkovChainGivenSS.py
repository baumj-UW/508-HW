'''
Created on May 7, 2019

@author: baumj

Given Steady State Markov
Find an irreducible markov chain that creates it
'''

import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

ss = np.array([[3/25, 3/25, 5/25, 5/25, 9/25]])

Pss = np.array([[22/25, 3/25, 0, 0, 0],\
               [3/25, 17/25, 5/25, 0, 0],\
               [0, 3/25, 17/25, 5/25, 0],\
               [0, 0, 5/25, 11/25, 9/25],\
               [0, 0, 0, 5/25, 20/25]])

Pss_trivial = np.array([[3/25, 3/25, 5/25, 5/25, 9/25],\
               [3/25, 3/25, 5/25, 5/25, 9/25],\
               [3/25, 3/25, 5/25, 5/25, 9/25],\
               [3/25, 3/25, 5/25, 5/25, 9/25],\
               [3/25, 3/25, 5/25, 5/25, 9/25]])

##Solve for SS using linear equation
J = np.eye(Pss.shape[0],Pss.shape[1]) - Pss.T
M = np.insert(J,5,1,axis=0)
b = np.zeros((6,1))
b[5,0] = 1
px = lstsq(M,b,rcond=None)[0]

print(px)

def MakeP(ss):
    ## generate P transition matrix given transient probabilities
    P = np.zeros((len(ss),len(ss)))
    for i in range(0,len(ss)-1):
        P[i,i+1] = ss[i+1]
    for i in range(1,len(ss)):
        P[i,i-1] = ss[i-1]
    for i in range(1,len(ss)-1):
        P[i,i] = 1 - (ss[i-1] + ss[i+1])
    P[0,0] = 1 - ss[1]
    P[len(ss)-1,len(ss)-1] = 1 - ss[len(ss)-2]
    return P

test = MakeP(ss[0])


print(test)