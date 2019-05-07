'''
Created on Apr 17, 2019

@author: baumj

EE 508 - HW2
'''

import numpy as np
from prettytable import PrettyTable

#Transition Probability Matrix
P = np.array([[0.12,0.71,0.15,0.02],\
             [0.15,0.4,0.4,0.05],\
             [0.05,0.45,0.3,0.2],\
             [0.85,0.1,0.04,0.01]])

alpha = np.array([1,0,0,0]) #initial state

#print multistep distribution
res = PrettyTable()
res.field_names = ["Step","P(S0)","P(S1)","P(S2)","P(S3)"]

for i in range(21):
    vals = np.matmul(alpha,np.linalg.matrix_power(P,i))
    row = vals.tolist()
    row.insert(0,i)
    res.add_row(row)
   
print(res)
print("working?")