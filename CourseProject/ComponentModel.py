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
from scipy.integrate import solve_ivp #ODE45 
import matplotlib.pyplot as plt


## Q matrix based on HV Air-Blast Circuit Breakers (historical data)
## 13 States [D1, D2, D3, F, I1, I2, I3, M1, MM1, M2, MM2, M3, MM3]
## Matrix from textbook doesn't align with the given steady state results...
# Q = np.array([[-0.01,0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0],\
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
#               [3.044E-04, 0, 0, -3.044E-04, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
#               [90.0, 0, 0, 0, -100.0, 0, 0, 5.0, 5.0, 0, 0, 0, 0],\
#               [0, 0, 0, 0, 0, -100.0, 0, 0, 0, 90.0, 10.0, 0, 0],\
#               [0, 0, 0, 0, 0, 0, -100.0, 0, 0, 0, 0, 10.0, 90.0],\
#               [99.0, 1.0, 0, 0, 0, 0, 0, -100.0, 0, 0, 0, 0, 0],\
#               [19.8, 0.2, 0, 0, 0, 0, 0, 0, -20.0, 0, 0, 0, 0],\
#               [50.0, 44.0, 1.0, 0, 0, 0, 0, 0, 0, -95.0, 0, 0, 0],\
#               [18.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, -20.0, 0, 0],\
#               [0, 10.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, -100.0, 80.0],\
#               [18.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -20.0]])
# steady state results from text:
#[0.542 0.219 0.043 0.195 0 0 0 0 0 0 0 0 0]

#Read Q matrix from csv 
#current file based on Endrenyi '98 paper
filepath = "C:/Users/baumj/Documents/UW Courses/EE 508 - Stochastic Processes/Repo/508-HW/CourseProject/CircuitBreakerQMatrix.csv"
Q = np.genfromtxt(filepath, delimiter=',')


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

#According to notes:
#P[t] = np.matmul(P[0],np.exp(Q*t))

## Try solving with ODE

def CTMC(t, x, Q):
    #x = np.exp(Q*t)
    dxdt = np.matmul(x,Q)
    return dxdt

STEP1 = 1000
SUB_INT = 500
eval_times = np.linspace(0,STEP1,SUB_INT)
                       
P0 = np.zeros(Q.shape[0])
P0[0] = 1 # set initial state as D1 (new)
results = solve_ivp(lambda t, x: CTMC(t, x, Q),\
                    [0,STEP1],P0,t_eval=eval_times)  

print('Results at final time step:')
print(results.y[:,-1])


## Test sample system
sampleResults = solve_ivp(lambda t, x:CTMC(t,x,sample),[0,STEP1],[1,0,0])
print('Results at final time step:')
print(sampleResults.y[:,-1])

print('working?')