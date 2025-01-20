# Manual eigenvalues and eigenvectors computation
def compute_eigenvalues_and_vectors(matrix, tol=1e-9, max_iter=5):
    """
    Compute eigenvalues and eigenvectors manually using the Power Iteration method.
    """
    n = len(matrix)  # Number of rows/columns (matrix is square)
    eigenvalues = []  # List to store eigenvalues
    eigenvectors = []  # List to store eigenvectors

    # Helper function: Normalize a vector (convert to unit vector)
    def normalize(vec):
        # Compute the magnitude (sqrt of sum of squares)
        norm = 0
        for x in vec:
            norm += x**2
        norm = norm**0.5

        # Divide each element by the magnitude
        normalized_vec = []
        for x in vec:
            normalized_vec.append(x / norm)
        return normalized_vec

    # Loop to calculate all eigenvalues/eigenvectors
    for _ in range(n):
        # Start with a random vector b (initial guess, all 1s)
        print("----------------------------------------------------------------n-------------------------------------------------------------")
        print(n)
        print([1.0] * n)
        b = [1.0] * n  

        for _ in range(max_iter):  # Power iteration loop
            # Multiply matrix by vector b
            b_next = []
            for i in range(n):
                row_sum = 0
                for j in range(n):
                    print("matrix[i][j] element : ", matrix[i][j])
                    print("b at j position : ", b[j])
                    row_sum += matrix[i][j] * b[j]
                b_next.append(row_sum)
                print("new b_next : ", b_next)

            # Normalize b_next to prevent overflow
            b_next = normalize(b_next)

            # Check convergence: if b_next and b are "close enough," stop iterating
            converged = True
            for i in range(n):
                if abs(b_next[i] - b[i]) >= tol:
                    converged = False
                    break
            if converged:
                break

            # Update b to b_next for the next iteration
            b = b_next

        # Calculate the eigenvalue associated with the eigenvector b
        eigenvalue = 0
        for i in range(n):
            dot_product = 0
            for j in range(n):
                dot_product += matrix[i][j] * b[j]
            eigenvalue += b[i] * dot_product

        eigenvalues.append(eigenvalue)  # Store eigenvalue
        eigenvectors.append(b)  # Store eigenvector

        # Deflate the matrix: subtract the found eigenvalue/vector influence
        new_matrix = []
        for i in range(n):
            new_row = []
            for j in range(n):
                new_value = matrix[i][j] - eigenvalue * b[i] * b[j]
                new_row.append(new_value)
            new_matrix.append(new_row)
        matrix = new_matrix

    return eigenvalues, eigenvectors

# Example: Compute eigenvalues and eigenvectors of a covariance matrix
cov_mat = [
    [1, 0.9, 0.8],
    [0.9, 1, 0.85],
    [0.8, 0.85, 1]
]

eigen_vals, eigen_vecs = compute_eigenvalues_and_vectors(cov_mat)

# Print results to understand the output
print("Eigenvalues:", eigen_vals)
print("Eigenvectors:", eigen_vecs)