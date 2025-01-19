
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)

# print(df_wine)

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
# standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np
cov_mat = np.cov(X_train_std.T)
print("cov_mat: ", cov_mat)

def compute_eigenvalues_and_vectors(matrix, tol=1e-9, max_iter=1000):
    """
    Compute eigenvalues and eigenvectors manually using the Power Iteration method.
    This is a simplified version to calculate the dominant eigenvalue and its eigenvector iteratively.
    """
    n = len(matrix)
    eigenvalues = []
    eigenvectors = []

    # A helper function to normalize a vector
    def normalize(vec):
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]

    for _ in range(n):  # Iterate for each eigenvalue/vector
        # Start with a random vector
        b = [1.0] * n
        for _ in range(max_iter):  # Power iteration to approximate eigenvalue/vector
            # Multiply matrix by vector
            b_next = [sum(matrix[i][j] * b[j] for j in range(n)) for i in range(n)]
            # Normalize the result
            b_next = normalize(b_next)
            # Check for convergence
            if all(abs(b_next[i] - b[i]) < tol for i in range(n)):
                break
            b = b_next

        # Approximate eigenvalue: (Av) / v
        eigenvalue = sum(b[i] * sum(matrix[i][j] * b[j] for j in range(n)) for i in range(n))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(b)

        # Deflate the matrix to find the next eigenvalue/vector
        matrix = [[matrix[i][j] - eigenvalue * b[i] * b[j] for j in range(n)] for i in range(n)]

    return eigenvalues, eigenvectors


# Example usage with the covariance matrix
eigen_vals, eigen_vecs = compute_eigenvalues_and_vectors(cov_mat)

print("\nEigenvalues (manual):\n", eigen_vals)
print("\nEigenvectors (manual):\n", eigen_vecs)
# print('\nEigenValues \n', eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in 
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
plt.bar(range(1,14), var_exp, align='center',
        label='Individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', 
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                eigen_pairs[1][1][:, np.newaxis]))

print('Matrix W:\n', w)