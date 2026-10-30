import numpy as np


class PowerIteration:
    """Find the largest eigenvalue and its eigenvector of a symmetric matrix."""

    def __init__(self, max_iterations=5000, tolerance=1e-10):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def find_top_eigenpair(self, matrix, seed=0):
        """Return (eigenvalue, eigenvector, number of steps used)."""
        # Step 1: start from a random unit vector
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(matrix.shape[0])
        vector = vector / np.linalg.norm(vector)

        for _ in range(self.max_iterations):
            # Step 2: multiply by the matrix
            next_vector = matrix @ vector
            length = np.linalg.norm(next_vector)
            if length == 0.0:
                return 0.0, vector  # no positive eigenvalue is left
            # Step 3: normalize to length 1
            next_vector = next_vector / length
            # Step 4: stop when the direction no longer changes
            if self.has_converged(vector, next_vector):
                vector = next_vector
                break
            vector = next_vector

        # Step 5: eigenvalue = Rayleigh quotient v^T B v
        eigenvalue = float(vector @ matrix @ vector)
        return eigenvalue, vector

    def has_converged(self, old_vector, new_vector):
        """True when new_vector = old_vector or -old_vector."""
        change = min(
            np.linalg.norm(new_vector - old_vector),
            np.linalg.norm(new_vector + old_vector),
        )
        return change < self.tolerance

    def trace_steps(self, matrix, start_vector, n_steps):
        """Run n_steps by hand; return a list of (step, B v, v, Rayleigh)."""
        vector = np.asarray(start_vector, dtype=float)
        history = []
        for step in range(1, n_steps + 1):
            product = matrix @ vector
            vector = product / np.linalg.norm(product)
            history.append((step, product, vector, vector @ matrix @ vector))
        return history
