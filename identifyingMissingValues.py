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