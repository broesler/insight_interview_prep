#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: run_kstest.py
#  Created: 07/08/2018, 19:02
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler
from scipy.stats import norm, uniform, kstest

np.random.seed(56)
plt.ion()

def make_kstest_plots(dist, compare='norm'):
    fig = plt.figure(1)
    plt.clf()
    ax = plt.gca()

    n = np.array([int(10**i) for i in range(7)])
    n = np.hstack((n, 3*n))
    n.sort()
    D = np.zeros(n.size)
    p = np.zeros(n.size)
    rvs = []
    for i in range(n.size):
        # Kolmogorov-Smirnov test if RVs drawn from normal distribution
        rvs.append(dist.rvs(size=n[i]))
        D[i], p[i] = kstest(rvs[i], compare)

    ax.plot(n, D, c='C3', label='D statistic')
    ax.plot(n, p, c='C0', label='p-value')
    ax.set_xscale('log')
    ax.legend()

    plt.figure(2, figsize=(11, 5))
    plt.clf()
    ax = plt.gca()
    ax.set_prop_cycle(cycler('color', 
                            [plt.cm.viridis(i) for i in np.linspace(0, 1, n.size)]))
    for i in range(n.size):
        sns.distplot(rvs[i], hist=False, ax=ax, label='n = {:2.2g}'.format(n[i]))
    ax.legend()

    plt.show(block=False)


# Define standard normal distribution
# dist = norm(loc=0, scale=1)
dist = uniform(loc=0, scale=1)
compare = 'norm'

make_kstest_plots(dist, compare)

#==============================================================================
#==============================================================================
