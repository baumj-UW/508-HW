'''
Created on Jun 5, 2019

@author: baumj

Contains constants and system configuration paramters 
'''
import numpy as np

Xfmr_Qfile = "TransformerQMatrix_abs.csv"
Cbkr_Qfile = "CircuitBreakerQMatrix_abs.csv"

NEW_XFMR = True
NEW_CBKR = True


YEARS = 200
SIM_START = 0
SIM_END = YEARS*365
SUB_INT = 10000#round(STEP1/(YEARS/4))
eval_times = np.linspace(SIM_START,SIM_END,SUB_INT)

CTMT_STATES = "[D1, D2, D3, F, I1, I2, I3, M1, MM1, M2, MM2, M3, MM3]"