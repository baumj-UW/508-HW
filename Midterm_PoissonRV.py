'''
Created on May 6, 2019

@author: baumj

Midterm Problem 1:
Show that the  normal distribution is a good
approximation of Poisson 
'''

import numpy as np
import matplotlib.pyplot as plt

## Generate 1000 samples of poisson RVs and normal RVs with same mean
## show histograms and plot

N = 1000
alpha = 50

pois = np.random.poisson(alpha,N)
norm = np.random.normal(alpha, np.sqrt(alpha),N)

plt.hist(pois, 20, normed=True, label='Poisson Samples')
plt.hist(norm, 20, normed=True, label='Gaussian Samples')
plt.legend()
plt.title('Sample distributions with mean=50')
plt.show()