import math

import numpy as np
from numpy.typing import NDArray

from .power_iteration import PowerIteration


class SVD:
    """SVD by power iteration and deflation on the Gram matrix A^T A."""

    U: NDArray
    Sigma: NDArray
    V: NDArray

    def __init__(
        self, matrix, rank_tolerance=1e-9, max_iterations=5000, n_components=math.inf
    ):
        self.A = np.asarray(matrix, dtype=float)
        self.rank_tolerance = rank_tolerance
        self.power_iteration = PowerIteration(max_iterations=max_iterations)
        self.n_components = n_components

        # Run the full pipeline and store the three component matrices.
        self.U, self.Sigma, self.V = self.compute_svd_steps()

    # ------------------------------------------------------------------
    # Step-by-step pipeline
    # ------------------------------------------------------------------
    def compute_svd_steps(self):
        """Execute the full SVD pipeline and return (U, sigma, Vt)."""
        m, n = self.A.shape
        max_components = min(m, n)

        if self.n_components is not None:
            max_components = min(max_components, self.n_components)

        # Step 1: Gram matrix B = A^T A (n x n, symmetric)
        gram_matrix = self.construct_gram_matrix()

        singular_values, left_vectors, right_vectors = [], [], []

        for _ in range(max_components):
            # Step 2: largest eigenpair (lambda_i, v_i) of the current B
            eigenvalue, v = self.power_iteration.find_top_eigenpair(gram_matrix)
            if eigenvalue <= 0.0:
                break

            # Step 3: singular value sigma_i = sqrt(lambda_i)
            sigma = self.derive_singular_value(eigenvalue)
            if self.rank_reached(sigma, singular_values):
                break

            # Step 4: left singular vector u_i = A v_i / sigma_i
            u = self.compute_left_singular_vector(v, sigma)

            singular_values.append(sigma)
            right_vectors.append(v)
            left_vectors.append(u)

            # Step 5: deflation, remove the found direction from B
            gram_matrix = self.deflate_matrix(gram_matrix, eigenvalue, v)

        # Step 6: assemble final U, sigma and V^T
        return self.assemble_final_svd(left_vectors, singular_values, right_vectors)

    # ------------------------------------------------------------------
    # Individual steps (one method per block in the diagram)
    # ------------------------------------------------------------------
    def construct_gram_matrix(self):
        """Step 1: Build the Gram matrix B = A^T A."""
        return self.A.T @ self.A

    def derive_singular_value(self, eigenvalue):
        """Step 3: Derive singular value sigma_i = sqrt(lambda_i)."""
        return np.sqrt(eigenvalue)

    def compute_left_singular_vector(self, v, sigma):
        """Step 4: Compute left singular vector u_i = A v_i / sigma_i."""
        return self.A @ v / sigma

    def deflate_matrix(self, gram_matrix, eigenvalue, v):
        """Step 5: Deflate the Gram matrix by the found eigenpair."""
        return gram_matrix - eigenvalue * np.outer(v, v)

    def assemble_final_svd(self, left_vectors, singular_values, right_vectors):
        """Step 6: Assemble final U, Sigma (2D) and V matrices."""

        U = np.column_stack(left_vectors)
        Vt = np.vstack(right_vectors)
        V = Vt.T

        Sigma = np.diag(singular_values)

        return U, Sigma, V

    def rank_reached(self, sigma, singular_values):
        """True when sigma is tiny compared with sigma_1 (rank of A reached)."""
        if not singular_values:
            return False
        return sigma < self.rank_tolerance * singular_values[0]
