#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: data_challenge_week5.py
#  Created: 07/03/2018, 16:54
#   Author: Bernie Roesler
#
"""
  Description: Week 5 Data Challenge
    -- What are the main factors that drive employee churn? Do they make sense?
       Explain your findings.
    -- What might you be able to do for the company to address employee churn,
       what would be follow-up actions? If you could add to this data set just
       one variable that could help explain employee churn, what would that be?
    -- Your output should be in the form a a jupyter notebook and pdf output of
       a jupyter notebook in which you specify your results and how you got them.
"""
# Questions:
#   Do they quit or retire?
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.close('all')

# Load the data
df = pd.read_csv('employee_retention_data.csv', header=0, 
                 parse_dates=['join_date', 'quit_date']) 

TODAY = pd.to_datetime('2015-12-13') # end date of data

df['employee_id'] = df.employee_id.astype('int64')

# Label the data
df['has_quit'] = ~df.quit_date.isnull()

df.quit_date.fillna(value=TODAY, inplace=True)
df['time_employed'] = (df.quit_date - df.join_date).dt.days

feat_cols = ['dept', 'seniority', 'salary', 'time_employed']

#------------------------------------------------------------------------------ 
#        Data Exploration
#------------------------------------------------------------------------------

# #------------------------------------------------------------------------------ 
# #        EDA Plots
# #------------------------------------------------------------------------------
# sns.pairplot(df, vars=feat_cols, hue='has_quit')
# plt.tight_layout()

# #------------------------------------------------------------------------------ 
# #        PCA
# #------------------------------------------------------------------------------
# # X = df[feat_cols]
# X = df[['seniority', 'salary', 'time_employed']]
# y = df.has_quit
#
# # dvs = pd.get_dummies(X.dept)
# # X = X.join(dvs).drop(columns='dept')
#
# # Normalize the data (some in inches, yds, $, etc.)
# X_std = StandardScaler().fit_transform(X)
#
# # Perform PCA decomposition to see n most important stats
# pca = PCA()
# Y = pca.fit_transform(X_std)
#
# # U, sigma, V = np.linalg.svd(X_std.T) # sigma is shape (m,) array, NOT matrix
# # explained_variance_ratio = sigma / sum(sigma)
# # Transform the data to the principle space
# # Y = X_std.dot(U)
# # Ys = pd.DataFrame(Y, index=X.index) # use range for columns
#
# sigma = pca.singular_values_
# explained_variance_ratio = pca.explained_variance_ratio_
#
# m = sigma.shape[0]  # number of singular values
#
# thresh = 0.8
# idx_sig = np.argmax(np.cumsum(explained_variance_ratio) >= thresh)
# sigma_allowed = sigma[:idx_sig]
#
# # Print explained variance in each PC
# print(pd.DataFrame(pca.components_, columns=X.columns))

# plt.figure(2)
# sns.heatmap(np.log(pca.inverse_transform(np.eye(X.shape[1]))), cmap='inferno')

#------------------------------------------------------------------------------
#        Plot Singular Values
#------------------------------------------------------------------------------
# plt.figure(3, figsize=(6.4, 4.8*2))
# plt.clf()
# ax1 = plt.subplot(211)
#
# # Label singular values
# ax1.plot(sigma, 'x', color='darkred')
# ax1.plot(sigma_allowed, 'x', color='darkgreen')
#
# ax1.axhline(sigma[idx_sig], linestyle='-.', color='black',   linewidth=1)
# ax1.axvline(idx_sig,        linestyle='-.', color='darkred', linewidth=1)
#
# ax1.set_yscale('log')
# ax1.grid(which='minor', axis='y')
#
# ax1.set_title('Singular Values of Covariance Matrix')
# # ax1.set_xlabel('index')
# plt.setp(ax1.get_xticklabels(), visible=False) # hide labels for subplot
# ax1.set_ylabel('$\sigma$')
# # ax1.set_xticks(range(0, m, np.floor(m/10)))
#
#
# #------------------------------------------------------------------------------ 
# #        Explained Variance Ratio
# #------------------------------------------------------------------------------
# ax = plt.subplot(212, sharex=ax1)
# ax.axvline(idx_sig, color='darkred', linestyle='-.', linewidth=1)
# ax.plot(range(m), np.cumsum(explained_variance_ratio), 
#         label='cumulative')
# ax.bar(range(m), explained_variance_ratio, 
#        align='center', alpha=0.5, label='individual')
# ax.set_xlabel('Principal Component')
# ax.set_ylabel('Explained Variance Ratio')
# ax.set_ylim([0, 1.05])
# ax.legend()
#
# plt.tight_layout()
#
# plt.show(block=False)

#==============================================================================
#==============================================================================
