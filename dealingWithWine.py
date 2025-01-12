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
print(lr.intercept_)
print(lr.coef_)