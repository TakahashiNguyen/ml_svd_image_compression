import numpy as np


class CompressionMetrics:

    @staticmethod
    def storage_size(k, m, n):
        """Storage required for rank-k SVD."""
        return k * (m + n + 1)

    @staticmethod
    def compression_ratio(k, m, n):
        """Compression ratio for a single channel."""
        stored = CompressionMetrics.storage_size(k, m, n)
        return (m * n) / stored if stored > 0 else float("inf")

    @staticmethod
    def total_energy(A):
        """Total Frobenius energy."""
        return np.sum(np.asarray(A, dtype=np.float64) ** 2)

    @staticmethod
    def energy_kept(singular_values, k, total_energy):
        """Percentage of energy retained by first k singular values."""
        singular_values = np.asarray(singular_values)
        k = min(max(k, 0), len(singular_values))

        if total_energy <= 0:
            return 0.0

        return np.sum(singular_values[:k] ** 2) / total_energy * 100

    @staticmethod
    def frobenius_error(A, A_k):
        """Frobenius reconstruction error."""
        A = np.asarray(A, dtype=np.float64)
        A_k = np.asarray(A_k, dtype=np.float64)

        if A.shape != A_k.shape:
            raise ValueError("A and A_k must have the same shape.")

        return np.linalg.norm(A - A_k, ord="fro")

    @staticmethod
    def relative_error(A, A_k):
        """Relative Frobenius error."""
        A = np.asarray(A, dtype=np.float64)
        A_k = np.asarray(A_k, dtype=np.float64)

        if A.shape != A_k.shape:
            raise ValueError("A and A_k must have the same shape.")

        norm_A = np.linalg.norm(A, ord="fro")

        if norm_A == 0:
            return 0.0 if np.array_equal(A, A_k) else float("inf")

        return np.linalg.norm(A - A_k, ord="fro") / norm_A

    @staticmethod
    def theory_error(singular_values, k):
        """Eckart-Young optimal Frobenius error."""
        singular_values = np.asarray(singular_values)
        k = min(max(k, 0), len(singular_values))

        return np.sqrt(np.sum(singular_values[k:] ** 2))

    @staticmethod
    def psnr(A, A_k, max_value=255.0):
        """Peak Signal-to-Noise Ratio."""
        A = np.asarray(A, dtype=np.float64)
        A_k = np.asarray(A_k, dtype=np.float64)

        if A.shape != A_k.shape:
            raise ValueError("A and A_k must have the same shape.")

        mse = np.mean((A - A_k) ** 2)

        if mse == 0:
            return float("inf")

        return 10 * np.log10(max_value**2 / mse)

    @staticmethod
    def ycbcr_storage_size(k_y, k_c, m, n):
        """Storage size for YCbCr 4:2:0."""
        m_c = (m + 1) // 2
        n_c = (n + 1) // 2

        size_y = CompressionMetrics.storage_size(k_y, m, n)
        size_cb = CompressionMetrics.storage_size(k_c, m_c, n_c)
        size_cr = CompressionMetrics.storage_size(k_c, m_c, n_c)

        return size_y + size_cb + size_cr

    @staticmethod
    def ycbcr_compression_ratio(k_y, k_c, m, n):
        """Compression ratio for 3-channel YCbCr image."""
        original_size = 3 * m * n
        stored_size = CompressionMetrics.ycbcr_storage_size(k_y, k_c, m, n)

        return original_size / stored_size if stored_size > 0 else float("inf")
