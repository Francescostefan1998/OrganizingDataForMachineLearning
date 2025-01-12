import pandas as pd
import numpy as np

df_wine = pd.read_csv('https://archive.ics.uci.edu/' 
                 'ml/machine-learning-databases/'
                 'wine/wine.data', header=None)


# print(df)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash', 
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids', 
                   'Nonflavanoid phenols', 
                   'Poanthocyanins', 
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

# print('Class labels', np.unique(df_wine['Class label']))

df_wine.head()

# print(df_wine.head())
# print("------Dataset with the colums named---------------")
# print(df_wine)

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# print("-----X train---------------------")
# print(X_train)
# print("-----y train---------------------")
# print(y_train)
# print("-----X test---------------------")
# print(X_test)
# print("-----y test---------------------")
# print(y_test)

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

ex = np.array([0, 1, 2, 3, 4, 5])
# print('standardized:', (ex - ex.mean()) / ex.std())
# print('normalized:', (ex - ex.min()) / (ex.max() -ex.min()))
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)


# Regularization
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1', solver = 'liblinear', multi_class = 'ovr')
lr=LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
# intercept correspond to the bias units
print(lr.intercept_)
# coef correspond to the weight valuse
print(lr.coef_)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label = df_wine.columns[column + 1], color = color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C(inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.28, 1.03), ncol=1, fancybox=True)
plt.show()