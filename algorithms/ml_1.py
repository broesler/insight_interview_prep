#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: ml_1.py
#  Created: 07/02/2018, 14:54
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data = datasets.load_iris()

X = data.data # (150,4)
y = data.target # (150,)

# X = pd.DataFrame(data=X, columns=data.feature_names)
# y = pd.DataFrame(y, index=X.index)

df = pd.DataFrame(data=np.hstack([X, y.reshape(-1,1)]), columns=(data.feature_names + ['label']))

lm = LogisticRegression()
nb = GaussianNB()
rf = RandomForestClassifier()

for clf in [lm, nb, rf]:
    clf.fit(X, y)
    pred_clf = clf.predict(X)
    print(classification_report(y, pred_clf))

# plt.figure(1)
sns.pairplot(df, hue='label')
plt.show()
#==============================================================================
#==============================================================================
