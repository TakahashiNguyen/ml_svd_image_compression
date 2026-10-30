import os

import numpy as np
from PIL import Image

from data.image_loader import ImageLoader
from svd.svd import SVD


def visualize_u_sigma_v():
    image_path = "input.jpg"
    if not os.path.exists(image_path):
        print(f"Không tìm thấy tệp {image_path}")
        return

    os.makedirs("results/svd_parts", exist_ok=True)
    print("Loading image...")
    Y, _, _ = ImageLoader().load_ycbcr(image_path)

    print("Computing SVD...")
    # Lấy số thành phần vừa phải để dễ trực quan hóa
    max_k = min(Y.shape)
    svd = SVD(Y, n_components=max_k)

    U, Sigma, V = svd.U, svd.Sigma, svd.V
    sigma_vals = np.diagonal(Sigma) if Sigma.ndim == 2 else Sigma

    # Hàm chuẩn hóa ma trận bất kỳ về [0, 255] để lưu thành ảnh PIL
    def normalize_to_image(matrix):
        m_min, m_max = matrix.min(), matrix.max()
        if m_max > m_min:
            norm = (matrix - m_min) / (m_max - m_min) * 255
        else:
            norm = np.zeros_like(matrix)
        return Image.fromarray(norm.astype(np.uint8))

    # 1. Trực quan hóa U (Left Singular Vectors - Biểu diễn đặc trưng dọc/không gian hàng)
    img_u = normalize_to_image(U)
    img_u.save("results/svd_parts/U_matrix.jpg")

    # 2. Trực quan hóa Sigma (Singular Values - Phân bố năng lượng đường chéo)
    # Vì Sigma là vector/ma trận 1 chiều chứa các giá trị kỳ dị giảm dần, ta tái hiện thành ảnh dạng dải sáng tối
    sigma_matrix = np.diag(sigma_vals) if Sigma.ndim == 1 else Sigma
    # Log scale hoặc chuẩn hóa để nhìn rõ sự chênh lệch lớn giữa sigma lớn và sigma nhỏ
    img_sigma = normalize_to_image(np.log1p(sigma_matrix))
    img_sigma = img_sigma.resize(
        (512, 512), Image.Resampling.NEAREST
    )  # Phóng to cho dễ nhìn
    img_sigma.save("results/svd_parts/Sigma_matrix.jpg")

    # 3. Trực quan hóa V^T (Right Singular Vectors transposed - Biểu diễn đặc trưng ngang/không gian cột)
    img_v = normalize_to_image(V)
    img_v.save("results/svd_parts/Vt_matrix.jpg")


if __name__ == "__main__":
    visualize_u_sigma_v()
