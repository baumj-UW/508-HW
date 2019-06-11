'''
Created on May 27, 2019

@author: Jackie Baum, Lane Smith
EE508 - Final Project
CT Markov Model

Component level model functions

Sample component based on G. Anders "Optimal Maintenance Policies for Power Equipment"
Section 4.3.2 (p32) and IEEE Goldbook Survey on Industrial power system equipment
'''
import numpy as np
from numpy.linalg import lstsq
from scipy.integrate import solve_ivp #ODE45 
import matplotlib.pyplot as plt


def SolveSS(Q):
    #Solve for stationary probabilities given Q matrix for CTMC
    M = np.insert(Q.T,Q.shape[0],1,axis=0)
    b = np.zeros((Q.shape[0]+1,1))
    b[-1,0] = 1
    px = lstsq(M,b,rcond=None)[0]
    return px[:,0] 


def CTMC(t, x, Q):
    dxdt = np.matmul(x,Q)
    return dxdt


def stateProb(Qfile,init_new,eval_times,start_t=0,end_t=10):
    '''
    Qfile = filepath of Q matrix
    Q = square generator matrix
    init_state = vector of initial state probabilities
    start_t = start time of simulation, default=0
    end_t = end time of simulation, default = 10
        simulation time unit must be the same as Q rates
        
    Returns state probabilities over time from Q
    '''
    Q = np.genfromtxt(Qfile, delimiter=',')
   
    init_state = np.ones(Q.shape[0])
    if init_new: 
        init_state[1:] = 0
    else:
        init_state = init_state/Q.shape[0]
        
    results = solve_ivp(lambda t, x: CTMC(t, x, Q),\
                        [start_t,end_t],init_state,t_eval=eval_times)      
    return results

## Test sample system
# sampleResults = solve_ivp(lambda t, x:CTMC(t,x,sample),[0,STEP1],[1,0,0])
# print('Sample Results at final time step:')
# print(sampleResults.y[:,-1])


## plot the results
def addPlots(results,figNum = 1,comp_name="Component"):
    '''
    Returns new figure which can be plotted with plt.show()
    '''    
    STATES = ['D1','D2','D3','F']
    stateprobfig = plt.figure(figNum)
    for (i,s) in enumerate(STATES):
        plt.plot(results.t,results.y[i],label=s)
    plt.grid(True)
    plt.xlabel('Time (days)')
    plt.ylabel('Prob of being in state')
    plt.legend()
    plt.title(comp_name + " State probabilities over time")
    return stateprobfig

