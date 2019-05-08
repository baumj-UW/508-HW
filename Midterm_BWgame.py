'''
Created on May 6, 2019

@author: baumj

EE 508 - Midterm Problems
'''
#from sympy import *
import numpy as np
from prettytable import PrettyTable

## Prob 2 - W/B 2 run game winner
#5 states: [tie, W1, B1, W2, B2]

#Transition Probability Matrix
P = np.array([[0.3,0.4,0.3,0, 0],\
             [0.25,0,0.25,0.5,0],\
             [0.2,0.3,0, 0,0.5],\
             [0, 0, 0, 1, 0],\
             [0, 0, 0, 0, 1]])

alpha = np.array([1,0,0,0,0]) #initial state

#print multistep distribution
res = PrettyTable()
res.field_names = ["Step","P(S0)","P(S1)","P(S2)","P(S3)","P(S4)"]

for i in range(0,101,10):
    vals = np.matmul(alpha,np.linalg.matrix_power(P,i))
    row = vals.tolist()
    row.insert(0,i)
    res.add_row(row)
   
print(res)

#t, w, b, ww, bb = symbols('t w b ww bb')
R = P[:3,3:]
Q = P[:3,:3]

N = np.linalg.inv(np.eye(3,3) - Q)
B = np.matmul(N,R)
#B 00 = p(W) win, B_01 = P(B) win steady state

c = np.ones((3,1))
t = np.matmul(N,c)
#t0 = E[# games until winner declared, starting from tie]

print("working?")