import numpy as np

# Define your matrix
A = np.array([[2, 1, 0],
              [1, 2, 1],
              [0, 1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
