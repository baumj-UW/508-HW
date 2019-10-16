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

#print(px)

## general functions## 

def SolveSS(P):
    #Solve for stationary probabilities given P matrix
    J = np.eye(P.shape[0],P.shape[1]) - P.T
    M = np.insert(J,P.shape[0],1,axis=0)
    b = np.zeros((P.shape[0]+1,1))
    b[P.shape[0],0] = 1
    px = lstsq(M,b,rcond=None)[0]
    return px[:,0]

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
check = SolveSS(test)


print("Generated P:",test)
print("Check stationary probabilities:",check)
print("original",ss[0])

test1 = np.array([[1/4, 1/4, 1/4, 1/4]])
test2 = np.array([[5/30, 10/30, 7/30, 3/30, 5/30]])
test3 = np.array([[2/10, 5/10, 3/10]])
test4 = np.array([[1/2, 1/4, 1/8, 1/16, 1/16]])
test5 = np.array([[3/25, 3/25, 5/25, 5/25, 9/25]])

print("Generalized Test:")
res1 = MakeP(test1[0])
check1 = SolveSS(res1)
print("Generated P:",res1)
print("Check stationary probabilities:",check1)
print("original",test1[0])
print("************************")
res2 = MakeP(test2[0])
check2 = SolveSS(res2)
print("Generated P:",res2)
print("Check stationary probabilities:",check2)
print("original",test2[0])
print("************************")
res3 = MakeP(test3[0])
check3 = SolveSS(res3)
print("Generated P:",res3)
print("Check stationary probabilities:",check3)
print("original",test3[0])
print("************************")
res4 = MakeP(test4[0])
check4 = SolveSS(res4)
print("Generated P:",res4)
print("Check stationary probabilities:",check4)
print("original",test4[0])
print("************************")
res5 = MakeP(test5[0])
check5 = SolveSS(res5)
print("Generated P:",res5)
print("Check stationary probabilities:",check5)
print("original",test5[0])
print("************************")
