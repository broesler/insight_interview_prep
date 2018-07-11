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
warnings.simplefilter('once', UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler # for plot colors
from scipy import stats

np.random.seed(56) # ensure reproducibility
plt.close('all')

# Globals
samples = int(5e3)
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
N = stats.norm(loc=0, scale=1)

## Define test distribution
# Uniform
# a = 1
# b = 9 # s.t. pdf = 1/8 = 0.125
# dist = stats.uniform(loc=a, scale=b-a)     # ~ dist[loc, loc+scale]

# Exponential
lam = 1
dist = stats.expon(scale=1/lam)
# scale = 1/lambda for f(x) = lambda * exp(-lambda*x)

#------------------------------------------------------------------------------ 
#        Plot the pdf of the test distribution
#------------------------------------------------------------------------------
# Draw samples on the range where pdf has support
x = np.linspace(dist.ppf(res), dist.ppf(1-res), 100)

fig = plt.figure(1)
fig.clf()
ax = plt.gca()
ax.set_title('Test Distribution')
ax.plot(x, dist.pdf(x), 'r-', lw=5, alpha=0.6, label='test pdf')

# Draw from the distribution and display the histogram
r = dist.rvs(size=int(1e3))
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, label='samples')
ax.legend(loc='lower right')

#------------------------------------------------------------------------------ 
#        Plots vs. n
#------------------------------------------------------------------------------
# Plot histogram of samples vs normal distribution
fig = plt.figure(2, figsize=(11,9))
fig.clf()

xN = np.linspace(N.ppf(res), N.ppf(1-res), 1000)

n_arr = [1, 2, 10, 30]
for i in range(len(n_arr)):
    # Convolve the pdfs
    Zn = convolve_dist(dist, n=n_arr[i], samples=samples)
    Nn = stats.gaussian_kde(Zn)   # compare to actual normal

    # Plot vs standard normal distribution
    ax = fig.add_subplot(2, 2, i+1)
    sns.distplot(Zn, kde=False, norm_hist=True, ax=ax)
    ax.plot(xN, Nn.pdf(xN), 'C0', label='$Z_n$ KDE')
    ax.plot(xN, N.pdf(xN), 'C3', label='$\mathcal{N}(0,1)$')
    ax.set_xlim([-4, 4])
    # ax.set_ylim([0, 1.25*max(Nn.pdf(xN))])
    ax.set_title("n = {}".format(n_arr[i]))

fig.suptitle(("Central Limit Theorem, $N_{{samples}}$ = {}\n" + \
             "$S_n = X_1 + \dots + X_n$").format(samples))
ax.legend(loc='lower right')

#------------------------------------------------------------------------------ 
#        Determine n to match N(0,1)
#------------------------------------------------------------------------------
# Draw from distrubution until we approach a standard normal
MAX_N = 100
thresh = 1 - 1e-3
score = np.inf
n = 1
D = np.empty(MAX_N)
W = np.empty(MAX_N)
A = np.empty(MAX_N)
K = np.empty(MAX_N)
p = np.empty(MAX_N)
X2 = np.empty(MAX_N)
D.fill(np.nan)
A.fill(np.nan)
W.fill(np.nan)
K.fill(np.nan)
p.fill(np.nan)
X2.fill(np.nan)
Zn = []
while n < MAX_N:
    # Compute convolution
    Zn.append(convolve_dist(dist, n=n, samples=samples))
    # Test if convolution is equivalent to normal distribution
    D[n], p[n] = stats.kstest(Zn[-1], 'norm')
    W[n], _ = stats.shapiro(Zn[-1])
    A[n], cv, sig = stats.anderson(Zn[-1], dist='norm')
    K[n], _ = stats.normaltest(Zn[-1])
    # # Chi-squared test requires bins of data
    # Zn_hist, _ = np.histogram(Zn[-1], bins=100, density=True)
    # N_hist, _ = np.histogram(N.rvs(size=100000), bins=100, density=True)
    # X2[n], _ = stats.chisquare(f_obs=Zn_hist, f_exp=N_hist)
    if W[n] > thresh:
        break
    n += 1

if n == MAX_N:
    print("Warning! MAX_N = {} reached!".format(MAX_N))

print("Results:\n\tn = {} for D = {}. p = {}".format(n, D[n-1], p[n-1]))

# Plot D and W test statistics
plt.figure(9)
plt.clf()
ax = plt.gca()
ax.plot(np.arange(MAX_N), D, c='C3', label='$D$ statistic')
ax.plot(np.arange(MAX_N), 1-W, c='C2', label='$W$ statistic')
# ax.plot(np.arange(MAX_N), X2/np.nanmax(X2), c='C4', label='$\chi^2$ statistic')
ax.plot(np.arange(MAX_N), K/np.nanmax(K), c='C1', label='$K^2$ statistic')
ax.plot(np.arange(MAX_N), p, c='C0', label='$p$-value')
# ax.set_yscale('log')
ax.set_title('Test Statistics vs. $n$')
ax.set_xlabel('Number of convolved distributions')
ax.set_ylabel('Statistic')
# ax.set_ylim([0, 2])
ax.legend(loc='upper right')

# Plot A^2 statistic (Anderson test)
# If A^2 is larger than critical value for corresponding significance level,
# the null hypothesis that the data come from the chosen distribution can be
# rejected
plt.figure(10)
plt.clf()
ax = plt.gca()
ax.plot(np.arange(MAX_N), A, c='C1', label='$A^2$ statistic')
# Use greys for threshold values
ax.set_prop_cycle(cycler('color',
                         [plt.cm.bone(i) for i in np.linspace(0, 0.75, 5)]))
for i in range(5):
    ax.plot(np.array([0, n]), cv[i]*np.array([1, 1]),
            label='Threshold {}%'.format(sig[i]))
ax.set_yscale('log')
ax.set_title('Test Statistics vs. $n$')
ax.set_xlabel('Number of convolved distributions')
ax.set_ylabel('Statistic')
ax.legend(loc='upper right')

# Q-Q plot
plt.figure(11)
plt.clf()
ax = plt.gca()
colors = [plt.cm.bone(i) for i in np.linspace(0, 0.9, len(Zn))][::-1]
for i in range(len(Zn)):
    result = stats.probplot(Zn[i], plot=ax)
    ax.get_lines()[2*i].set_markeredgecolor('none')
    ax.get_lines()[2*i].set_markerfacecolor(colors[i])
    # Turn off all but last fit line
    if i < len(Zn)-1:
        ax.get_lines()[2*i+1].set_linestyle('none')

plt.show()
#==============================================================================
#==============================================================================
