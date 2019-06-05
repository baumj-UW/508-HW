'''
Created on May 30, 2019

@author: Jackie Baum, Lane Smith
EE508 - Final Project
CT Markov Model

System Model probability based on Ross - Reliability Chapter
'''

import numpy as np
#from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from CourseProject.ComponentModel import stateProb, addPlots
from CourseProject.sys_config import *


##reliability function
#try 3 components in parallel + 1 in series
# P = np.zeros((4,len(results.t)))
# for comp in range(4):
#     P[comp] = results.y[3] #prob that device is in fail state (consider including other states like maintenance?)

Pxfmr = stateProb(Xfmr_Qfile,NEW_XFMR,eval_times,start_t=SIM_START,end_t=SIM_END)
Pcbkr = stateProb(Cbkr_Qfile,NEW_CBKR,eval_times,start_t=SIM_START,end_t=SIM_END)

#Plots of component probabilities
addPlots(Pxfmr,1,"Transformer") 
addPlots(Pcbkr,2,"Circuit Breaker")


P = [Pxfmr.y[3]]*1 + [Pcbkr.y[3]]*3 #failure probs of xfmr in series with 3 parallel cbks
results = Pxfmr ##just to test
#make sure this is elementwise multiplication
R = (1-P[0])*(1-(P[1])*(P[2])*(P[3])) #from Ross Reliability chapter, p(working)=1-P(fail) for each component 

#Determine the mean time to failure (MTTF) of the system
MTTF_array = []
for i in range(len(R)):
    if i > 0:
        #Compute MTTF using numerical integration (trapezoidal rule) for each segment
        MTTF_elem = (1/2)*(R[i] + R[i - 1])*(results.t[i] - results.t[i - 1])
        MTTF_array.append(MTTF_elem)

#Sum all the numerically-integrated segments for the total MTTF
MTTF = sum(MTTF_array)/365
print("The system mean time to failure is %.2f years.\n" % MTTF)

#Reliability of one component
R1 = (1 - P[0])

#Determine the mean time to failure (MTTF) of a single element
MTTF_array1 = []
for i in range(len(R1)):
    if i > 0:
        #Compute MTTF using numerical integration (trapezoidal rule) for each segment
        MTTF_elem1 = (1/2)*(R1[i] + R1[i - 1])*(results.t[i] - results.t[i - 1])
        MTTF_array1.append(MTTF_elem1)

#Sum all the numerically-integrated segments for the total MTTF
MTTF1 = sum(MTTF_array1)/365
print("The component mean time to failure is %.2f years.\n" % MTTF1)

relfig = plt.figure(3) #fig num
plt.plot(results.t,R)
plt.grid(True)
plt.xlabel('Time (days)')
plt.ylabel('Probability that System is Functioning')
plt.title("System Reliability")

plt.show()
print('working?')