import numpy as np

from svd.svd import SVD


class SVDProcessor:
    """
    Thực hiện phân rã SVD và tái tạo ma trận
    với số lượng singular values được giữ lại.
    """

    def __init__(self, rank_tolerance=1e-3, max_iterations=5000):
        self.rank_tolerance = rank_tolerance
        self.max_iterations = max_iterations

    def decompose(self, matrix, max_rank):
        """
        Phân rã ma trận bằng SVD.

        A = U @ Sigma @ V.T
        """
        return SVD(
            matrix,
            rank_tolerance=self.rank_tolerance,
            max_iterations=self.max_iterations,
            n_components=max_rank,
        )

    @staticmethod
    def reconstruct(svd, k):
        """
        Tái tạo ma trận từ k singular values lớn nhất.

        A_k = U_k @ Sigma_k @ V_k.T
        """
        k = min(k, len(svd.Sigma))

        if svd.Sigma.ndim == 1:
            sigma = np.diag(svd.Sigma[:k])
        else:
            sigma = svd.Sigma[:k, :k]

        return svd.U[:, :k] @ sigma @ svd.V[:, :k].T

    @staticmethod
    def get_singular_values(svd):
        """Lấy danh sách singular values."""
        if svd.Sigma.ndim == 2:
            return np.diagonal(svd.Sigma)

        return svd.Sigma
