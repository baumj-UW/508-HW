'''
Created on May 30, 2019

@author: Jackie Baum, Lane Smith
EE508 - Final Project
CT Markov Model

System Model probability based on Ross - Reliability Chapter
'''

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from CourseProject.ComponentModel import results


##reliability function
#try 3 components in parallel + 1 in series
P = np.zeros((4,len(results.t)))
for comp in range(4):
    P[comp] = results.y[3] #prob that device is in fail state (consider including other states like maintenance?)

#make sure this is elementwise multiplication
R = (1-P[0])*(1-(P[1])*(P[2])*(P[3])) #from Ross Reliability chapter, p(working)=1-P(fail) for each component 

relfig = plt.figure(2)
plt.plot(results.t,R)
plt.grid(True)
plt.xlabel('Time (days)')
plt.ylabel('Prob of system failure')
plt.title("System Reliability")

plt.show()
print('working?')