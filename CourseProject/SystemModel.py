'''
Created on May 30, 2019

@author: Jackie Baum, Lane Smith
EE508 - Final Project
CT Markov Model

System Model probability based on Ross - Reliability Chapter

Uses ComponentModel functions to calculate system level reliability 
'''

import numpy as np
import matplotlib.pyplot as plt
from CourseProject.ComponentModel import stateProb, addPlots
from CourseProject.sys_config import *

def MTTF_calc(R, t):
    MTTF_array = []
    for i in range(len(R)):
        if i > 0:
            #Compute MTTF using numerical integration (trapezoidal rule) for each segment
            MTTF_elem = (1/2)*(R[i] + R[i - 1])*(t[i] - t[i - 1])
            MTTF_array.append(MTTF_elem)
    
    #Sum all the numerically-integrated segments for the total MTTF
    MTTF = sum(MTTF_array)/365
    return MTTF


Pxfmr = stateProb(Xfmr_Qfile, NEW_XFMR, eval_times, start_t=SIM_START, end_t=SIM_END)
Pcbkr = stateProb(Cbkr_Qfile, NEW_CBKR, eval_times, start_t=SIM_START, end_t=SIM_END)

#Create plots of component probabilities
addPlots(Pxfmr,1,"Transformer") 
addPlots(Pcbkr,2,"Circuit Breaker")

#Copy component failure probabilities into a list
P = [Pxfmr.y[3]]*1 + [Pcbkr.y[3]]*3 #failure probs of xfmr in series with 3 parallel cbks


#Reliability functions; from Ross Reliability chapter, p(working)=1-P(fail) for each component 
R = (1 - P[0])*(1 - P[1]*P[2]*P[3]) #full system
R1 = (1 - P[0]) #single transformer
R2 = (1 - P[1]) #single circuit breaker

#Determine the mean time to failure (MTTF) of the system
MTTF = MTTF_calc(R, eval_times)
print("The system mean time to failure is %.2f years.\n" % MTTF)

#Determine the mean time to failure (MTTF) of a single transformer
MTTF1 = MTTF_calc(R1, eval_times)
print("The transformer mean time to failure is %.2f years.\n" % MTTF1)

#Determine the mean time to failure (MTTF) of a single circuit breaker
MTTF2 = MTTF_calc(R2, eval_times)
print("The circuit breaker mean time to failure is %.2f years.\n" % MTTF2)

relfig = plt.figure(3) #fig num
plt.plot(eval_times, R)
plt.grid(True)
plt.xlabel('Time (days)')
plt.ylabel('Probability that System is Functioning')
plt.title("System Reliability")

plt.show()