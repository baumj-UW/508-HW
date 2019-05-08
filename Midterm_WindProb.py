'''
Created on May 6, 2019

@author: baumj

Stationary Markov Chain model calculations
given P transition matrix from Wind Power Paper
'''

import numpy as np
from numpy.linalg import *
from prettytable import PrettyTable,MSWORD_FRIENDLY, PLAIN_COLUMNS
import matplotlib.pyplot as plt

P = np.array([[0.371,0.407,0.174,0.036,0.009,0.001,0.001,0.001,0,0,0,0],\
              [0.166,0.446,0.312,0.059,0.012,0.004,0,0.001,0,0,0,0],\
              [0.051,0.243,0.504,0.163,0.028,0.008,0.002,0.001,0,0,0,0],\
              [0.017,0.083,0.303,0.391,0.16,0.035,0.008,0.002,0.001,0,0,0],\
              [0.01,0.035,0.099,0.277,0.382,0.157,0.031,0.007,0.001,0.001,0,0],\
              [0.006,0.021,0.043,0.108,0.295,0.343,0.146,0.031,0.004,0.003,0,0],\
              [0.005,0.016,0.027,0.047,0.11,0.302,0.324,0.142,0.021,0.004,0.002,0],\
              [0.006,0.016,0.03,0.033,0.055,0.127,0.365,0.239,0.105,0.022,0.002,0],\
              [0.009,0.019,0.014,0.018,0.042,0.065,0.14,0.326,0.269,0.079,0.014,0.005],\
              [0.014,0.054,0.055,0.014,0.027,0.028,0.041,0.205,0.288,0.164,0.083,0.027],\
              [0,0,0,0.04,0,0,0.08,0.12,0.16,0.24,0.28,0.08],\
              [0,0,0,0,0,0,0,0.2,0,0.2,0.6,0]])


##Solve for SS iteratively 
alpha = np.zeros((1,12)) #[1,0,0,0,0]) #initial state
alpha[0,0] = 1
#print multistep distribution
res = PrettyTable()
res.field_names = ["Step","P(S0)","P(S1)","P(S2)","P(S3)","P(S4)", "5", "6", "7", "8", "9","10","11"]
res.set_style(MSWORD_FRIENDLY)

for i in range(0,51,5):
    vals = np.matmul(alpha[0],matrix_power(P,i))
    row = vals.tolist()
    row.insert(0,i)
    res.add_row(row)

print("Steady State distribution by iterating from S0")   
print(res)


##Solve for SS using linear equation
J = np.eye(P.shape[0],P.shape[1]) - P.T
M = np.insert(J,12,1,axis=0)
b = np.zeros((13,1))
b[12,0] = 1
px = lstsq(M,b,rcond=None)[0]

print("Steady State distribution solving a linear equation:")
print(px)

speed = np.linspace(1,12,12)
markerline, stemlines, baseline = plt.stem(speed, px, 'b-', basefmt='k-')  
plt.xticks(speed)
plt.xlabel('Speed (m/s)')
plt.ylabel('Probability')
plt.title("Steady-State PMF of Wind Speeds")
plt.show()

mean = (np.dot((speed-0.5),px))
var = np.dot((speed-0.5)**2,px) - mean**2

print("PMF statistics:")
print("Mean:",mean)
print("Variance:",var)
print("Standard Deviation:",np.sqrt(var))


