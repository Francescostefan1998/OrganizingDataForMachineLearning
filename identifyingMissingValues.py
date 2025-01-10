import pandas as pd
import numpy as np

df = pd.DataFrame([[1, 2, 3, 4],
                   [5, 6, np.nan, 8],
                   [9, 10, np.nan, 12],
                   [np.nan, 14, 15, 16]],
                  dtype=float)

# Print the original DataFrame
print("Original DataFrame:")
print(df)

# Print after dropping rows with NaN
print("\nDataFrame after dropping rows with NaN:")
print(df.dropna())

# Print after dropping columns with NaN
print("\nDataFrame after dropping columns with NaN:")
print(df.dropna(axis=1))

# Print after dropping rows where all values are NaN
print("\nDataFrame after dropping rows where all values are NaN:")
print(df.dropna(how='all'))

print("------------------------------------------")

print(df.isnull().sum())

print("------------------------------------------")
print(df.values)

print("Drop rows that have fewer than 4 real values")
print(df.dropna(thresh=4))

print("Only drop rows where NaN appear in specific columns (here is 2)")
print(df.dropna(subset=[2]))


print("------------------------------------------")

from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

print("------------------------------------------")
print(df.fillna(df.mean()))

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.5, 'class2']])
                
df.columns = ['color', 'size', 'price', 'classlabel']
print("------------------------------------------")
print(df)

print("------------------------------------------")
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)

print(df)
print("------------------------------------------")

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

print(df['size'].map(inv_size_mapping))


import numpy as np
class_mapping = {label: idx for idx, label in
                enumerate(np.unique(df['classlabel']))}

print("----------------------------------")
print(class_mapping)

# now we use the mapping dictionary to transform the class labels into integers
print("---------------------------------")
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# now we reverse back to the original setup
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print("--------------------------------")
print(df)

# labelEncoder library 
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print("--------------------------------")
class_le.inverse_transform(y)
print(class_le.inverse_transform(y))

# Performing the hot encoding on nominal features
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
print("----------------OneHotEncoder----------------")
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())
print("--------------------------------")