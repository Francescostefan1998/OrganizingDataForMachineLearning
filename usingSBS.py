from sklearn.base import clone
from itertools import combinations
import numpy as np
from sbs import SBS
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df_wine = pd.read_csv('https://archive.ics.uci.edu/' 
                 'ml/machine-learning-databases/'
                 'wine/wine.data', header=None)



df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash', 
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids', 
                   'Nonflavanoid phenols', 
                   'Poanthocyanins', 
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']


df_wine.head()

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# from sklearn.preprocessing import MinMaxScaler
# mms = MinMaxScaler()
# X_train_norm = mms.fit_transform(X_train)
# X_test_norm = mms.fit_transform(X_test)

# ex = np.array([0, 1, 2, 3, 4, 5])

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)
# print(X_train_std)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
