'''
Created on May 30, 2019

@author: Jackie Baum, Lane Smith
EE508 - Final Project
CT Markov Model

System Model probability based on Ross - Reliability Chapter

Solves for various system probabilities and metrics
'''

import numpy as np
from scipy.integrate import solve_ivp #ODE45
import matplotlib.pyplot as plt



def CTMC(t, x, Q):
    #Presents the continuous-time Markov chain calculation in the Kolmogorov forward equation
    dxdt = np.matmul(x, Q)
    return dxdt

def MTTF_calc(R, t):
    #Calculates the mean time to failure (MTTF)
    MTTF_array = []
    for i in range(len(R)):
        if i > 0:
            #Compute MTTF using numerical integration (trapezoidal rule) for each segment
            MTTF_elem = (1/2)*(R[i] + R[i - 1])*(t[i] - t[i - 1])
            MTTF_array.append(MTTF_elem)
            
    #Sum all the numerically-integrated segments for the total MTTF
    MTTF = sum(MTTF_array)/365
    return MTTF

def KFE_solver(Q, init_state):
    #Solves the Kolmogorov forward equation problem
    #ODE-solver parameters
    STEP = 365*n_years #total number of days
    SUB_INT = 10000
    eval_times = np.linspace(0, STEP, SUB_INT)
    P0 = np.zeros(Q.shape[0])
    P0[init_state] = 1
    
    #Solve the Kolmogorov forward equation
    results = solve_ivp(lambda t, x: CTMC(t, x, Q), [0, STEP], P0, t_eval = eval_times)
    return results



#Select stationary probabilities (1), mean time to failure (2), or system structure analysis (3)
selector = 3

if selector == 1:
    #Read Q matrix from csv; current file based on Endrenyi '98 paper
    Q1 = np.genfromtxt('TransformerQMatrix.csv', delimiter = ',')
    Q2 = np.genfromtxt('CircuitBreakerQMatrix.csv', delimiter = ',')
    Q3 = np.genfromtxt('LQCircuitBreakerQMatrix_abs.csv', delimiter = ',')
    
    #Solve the Kolmogorov forward equation for the different components
    n_years = 100
    init_state = 0 #start in D1
    Pxfmr = KFE_solver(Q1, init_state)
    Pcbkr = KFE_solver(Q2, init_state)
    Pcbkr_lq = KFE_solver(Q3, init_state)
    
    #Combine the probabilities computed above into a single variable
    P = [Pxfmr.y[3]]*1 + [Pcbkr.y[3]]*3 + [Pcbkr_lq.y[3]]
    results = Pxfmr
    
    #A transformer is in series with a parallel combination of 3 circuit breakers
    R = (1 - P[0])*(1 - P[1]*P[2]*P[3]) 
    R1 = (1 - P[0]) #Single transformer
    R2 = (1 - P[1]) #Single circuit breaker
    R3 = (1 - P[4]) #Single circuit breaker
    
    #Determine the mean time to failure (MTTF) of the system
    MTTF = MTTF_calc(R, results.t)
    MTTF1 = MTTF_calc(R1, results.t)
    MTTF2 = MTTF_calc(R2, results.t)
    MTTF3 = MTTF_calc(R3, results.t)
    
    print("The system's mean time to failure is %.2f years.\n" % MTTF)
    print("The transformer's mean time to failure is %.2f years.\n" % MTTF1)
    print("The circuit breaker's mean time to failure is %.2f years.\n" % MTTF2)
    print("The lower-quality circuit breaker's mean time to failure is %.2f years.\n" % MTTF3)
    
    #Plot the reliability function of the system
    plt.figure()
    plt.plot(results.t, R)
    plt.grid(True)
    plt.xlabel('Time (Days)')
    plt.ylabel('Probability that System is Functioning')
    plt.title("System Reliability")
    plt.show()
    
elif selector == 2:
    #Read Q matrix from csv; current file based on Endrenyi '98 paper
    Q1 = np.genfromtxt('TransformerQMatrix.csv', delimiter = ',')
    Q2 = np.genfromtxt('CircuitBreakerQMatrix.csv', delimiter = ',')
    
    #Solve the Kolmogorov forward equation for the different components
    n_years = 40
    init_state = 0 #start in D1
    results = KFE_solver(Q1, init_state)
    
    #Plot the stationary probabilities
    STATES = ['D1', 'D2', 'D3', 'F']
    plt.figure()
    for (i, s) in enumerate(STATES):
        plt.plot(results.t, results.y[i], label = s)
    plt.grid(True)
    plt.xlabel('Time (Days)')
    plt.ylabel('Probability in State')
    plt.title("State Probabilities over Time")
    plt.legend()
    plt.show()
    
elif selector == 3:
    #Read Q matrix from csv; current file based on Endrenyi '98 paper
    Q = np.genfromtxt('CircuitBreakerQMatrix_abs.csv', delimiter = ',')
    
    #Solve the Kolmogorov forward equation for the different components
    n_years = 300
    init_state = 0 #start in D1
    #results = KFE_solver(Q, init_state)
    
    #Series-connected components
    S = np.zeros((100, len(results.t)))
    S_MTTF = []
    for i in range(S.shape[0]):
        S[i, :] = (1 - results.y[3])**(i + 1)
        S_MTTF.append(MTTF_calc(S[i, :], results.t))
    
    #Plot the results
    plt.figure()
    plt.plot(S_MTTF)
    plt.grid(True)
    plt.xlabel("Number of Series-Connected Components")
    plt.ylabel("MTTF (Years)")
    plt.title("Changes in MTTF for Series-Connected Components")
    
    #Parallel-connected components
    P = np.zeros((100, len(results.t)))
    P_MTTF = []
    for i in range(P.shape[0]):
        P[i, :] = 1 - (results.y[3])**(i + 1)
        P_MTTF.append(MTTF_calc(P[i, :], results.t))
    
    #Plot the results
    plt.figure()
    plt.plot(P_MTTF)
    plt.grid(True)
    plt.xlabel("Number of Parallel-Connected Components")
    plt.ylabel("MTTF (Years)")
    plt.title("Changes in MTTF for Parallel-Connected Components")

else:
    print("Not a valid option! Try again!")