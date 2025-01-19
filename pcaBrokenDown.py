# Creating a meaningful dataset directly
import numpy as np
import matplotlib.pyplot as plt

# Features: Age, Height, Weight, Heart Rate, Body Fat, Activity Level
# Labels: Fitness Level (1=Low, 2=Medium, 3=High)

# Dataset: 8 people with 6 features each
data = np.array([
    [25, 170, 70, 120, 0.8, 3.2, 1],  # Low fitness
    [30, 165, 68, 110, 0.9, 3.4, 2],  # Medium fitness
    [35, 180, 75, 130, 0.7, 3.1, 1],  # Low fitness
    [40, 175, 80, 140, 0.6, 3.5, 3],  # High fitness
    [28, 160, 72, 125, 0.85, 3.3, 1], # Low fitness
    [22, 155, 65, 100, 0.95, 3.6, 2], # Medium fitness
    [33, 185, 78, 135, 0.75, 3.0, 3], # High fitness
    [29, 168, 73, 115, 0.82, 3.4, 2], # Medium fitness
])

# Splitting features (X) and labels (y)
X = data[:, :-1]  # Features: Age, Height, Weight, Heart Rate, Body Fat, Activity Level
y = data[:, -1]   # Labels: Fitness Level (1, 2, 3)

# Splitting into training and test sets (manual split for simplicity)
train_ratio = 0.7  # 70% training data
split_idx = int(len(X) * train_ratio)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Feature standardization (manually implemented)
# Standardize = (value - mean) / std_dev
means = X_train.mean(axis=0)
std_devs = X_train.std(axis=0)

X_train_std = (X_train - means) / std_devs  # Standardize training features
X_test_std = (X_test - means) / std_devs   # Standardize test features

# Covariance matrix calculation
# Covariance measures how two features vary together
cov_mat = np.cov(X_train_std.T)

# Manual eigenvalues and eigenvectors computation
def compute_eigenvalues_and_vectors(matrix, tol=1e-9, max_iter=1000):
    """
    Compute eigenvalues and eigenvectors manually using the Power Iteration method.
    """
    n = len(matrix)
    eigenvalues = []
    eigenvectors = []

    # A helper function to normalize a vector
    def normalize(vec):
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]

    for _ in range(n):  # Iterate for each eigenvalue/vector
        b = [1.0] * n  # Start with a random vector
        for _ in range(max_iter):  # Power iteration
            b_next = [sum(matrix[i][j] * b[j] for j in range(n)) for i in range(n)]  # Multiply matrix by vector
            b_next = normalize(b_next)  # Normalize the result
            if all(abs(b_next[i] - b[i]) < tol for i in range(n)):  # Convergence check
                break
            b = b_next

        # Approximate eigenvalue
        eigenvalue = sum(b[i] * sum(matrix[i][j] * b[j] for j in range(n)) for i in range(n))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(b)

        # Deflate the matrix
        matrix = [[matrix[i][j] - eigenvalue * b[i] * b[j] for j in range(n)] for i in range(n)]

    return eigenvalues, eigenvectors

# Compute eigenvalues and eigenvectors
eigen_vals, eigen_vecs = compute_eigenvalues_and_vectors(cov_mat)

# # Explained variance (manual calculation)
# total_variance = sum(eigen_vals)
# var_exp = [(i / total_variance) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# # Plot explained variance
# plt.bar(range(1, len(var_exp) + 1), var_exp, align='center', label='Individual explained variance')
# plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='Cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

# # Print results
# print("Covariance Matrix:\n", cov_mat)
# print("\nEigenvalues:\n", eigen_vals)
# print("\nEigenvectors:\n", eigen_vecs)
