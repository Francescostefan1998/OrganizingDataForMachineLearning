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

print('Class labels', np.unique(df_wine['Class label']))

df_wine.head()

print(df_wine.head())
print("------Dataset with the colums named---------------")
print(df_wine)

