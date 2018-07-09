#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: central_limit_thm.py
#  Created: 07/08/2018, 14:55
#   Author: Bernie Roesler
#
"""
  Description: Example central limit theorem proofs
"""
#==============================================================================

import warnings
warnings.simplefilter('default', UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import uniform, norm, gaussian_kde, ttest_1samp, kstest

plt.ion()
np.random.seed(56) # ensure reproducibility
# plt.close('all')

# Globals
samples = int(1e4)
res = 1e-4

def convolve_dist(dist, n, samples=1000, norm_out=True):
    """Convolve a distribution n times.
    For a random variable X, 
        Sn = X1 + X2 + ... + Xn

    Parameters
    ----------
    dist : rv_continuous
        continuous distrubution object, i.e. scipy.stats.norm
    n : int
        number of convolutions to make
    samples : int, optional, default=1000
        number of samples to draw for each convolution
    norm_out : boolean, optional, default=True
        normalize output to Z-score: (S - n*dist.mean()) / np.sqrt(n*dist.var())
    
    Returns
    -------
    out : ndarray, shape (samples,)
        if norm_out, out = Zn values, otherwise out = Sn values
    """
    Sn = np.zeros(samples)
    for i in range(n):
        # Draw from distribution and add to sum
        Sn += dist.rvs(size=samples)

    if norm_out:
        Zn = (Sn - n*dist.mean()) / np.sqrt(n*dist.var()) # normalize Sn
        return Zn
    else:
        return Sn

# Define standard normal distribution
N = norm(loc=0, scale=1)

# Define test distribution
a = 1
b = 9 # s.t. pdf = 1/8 = 0.125
U = uniform(loc=a, scale=b-a)     # ~ U[loc, loc+scale]

#------------------------------------------------------------------------------ 
#        Plot the pdf of the test distribution
#------------------------------------------------------------------------------
# Draw samples on the range where pdf has support
x = np.linspace(U.ppf(res), U.ppf(1-res), 100)

fig = plt.figure(1)
fig.clf()
ax = plt.gca()
ax.set_title('Uniform Distribution')
ax.plot(x, U.pdf(x), 'r-', lw=5, alpha=0.6, label='uniform pdf')

# Draw from the distribution and display the histogram
r = U.rvs(size=int(1e3))
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, label='samples')
ax.legend(loc='lower right')

#------------------------------------------------------------------------------ 
#        Determine n to match N(0,1)
#------------------------------------------------------------------------------
# Draw from distrubution until we approach a standard normal
MAX_N = 100
thresh = 1
score = np.inf
n = 1
while n < MAX_N:
    # Compute convolution
    Zn = convolve_dist(U, n=n, samples=samples)
    # Test if convolution is equivalent to normal distribution
    kstest
    score = 0
    if score < thresh:
        break
    # Increase n
    n += 1

#------------------------------------------------------------------------------ 
#        Plots vs. n
#------------------------------------------------------------------------------
# Plot histogram of samples vs normal distribution
fig = plt.figure(2, figsize=(11,9))
fig.clf()

xN = np.linspace(N.ppf(res), N.ppf(1-res), samples)

n_arr = [1, 2, 10, 30]
for i in range(len(n_arr)):
    # Convolve the pdfs
    Zn = convolve_dist(U, n=n_arr[i], samples=samples)
    Nn = gaussian_kde(Zn)   # compare to actual normal

    # Plot vs standard normal distribution
    ax = fig.add_subplot(2, 2, i+1)
    sns.distplot(Zn, kde=False, norm_hist=True, ax=ax)
    ax.plot(xN, Nn.pdf(xN), 'C0', label='$Z_n$ KDE')
    ax.plot(xN, N.pdf(xN), 'C3', label='$\mathcal{N}(0,1)$')
    # ax.set_ylim([0, 1.25*max(Nn.pdf(xN))])
    ax.set_title("n = {}".format(n_arr[i]))

fig.suptitle('Central Limit Theorem Demonstration')
ax.legend(loc='lower right')

# plt.show(block=False)
#==============================================================================
#==============================================================================
