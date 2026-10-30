import os

import numpy as np
from PIL import Image
from tabulate import tabulate

from data.image_loader import ImageLoader
from metrics import CompressionMetrics
from svd.svd import SVD


def reconstruct(svd, k):
    k = min(k, len(svd.Sigma))
    sigma = np.diag(svd.Sigma[:k]) if svd.Sigma.ndim == 1 else svd.Sigma[:k, :k]
    return svd.U[:, :k] @ sigma @ svd.V[:, :k].T


def run_ycbcr_compression_table():
    image_path = "input.jpg"

    if not os.path.exists(image_path):
        print(f"Không tìm thấy tệp {image_path}")
        return

    print("Loading image...")
    Y, Cb, Cr = ImageLoader().load_ycbcr(image_path)
    m, n = Y.shape

    os.makedirs("results", exist_ok=True)

    # 1. Cấu hình danh sách dạng tuple (k_y, k_c)
    # k_y: Hạng kênh sáng Y (giữ chi tiết/đường nét)
    # k_c: Hạng kênh màu Cb, Cr (mắt người ít nhạy cảm nên chọn k_c nhỏ hơn k_y)
    configs = [
        (30, 10),  # Nén cao (k_c = 1/3 k_y)
        (60, 20),  # Cấu hình tối ưu (Mắt nhìn đẹp, CR cao)
        (120, 40),  # Chất lượng cao
        (240, 80),  # Chi tiết rất cao
        (480, 160),  # Gần nguyên bản
    ]

    # Subsample chroma channels (4:2:0) -> Kích thước Cb, Cr là (m/2, n/2)
    channels = [Y, Cb[::2, ::2], Cr[::2, ::2]]

    # Lấy max rank riêng cho Y và Chroma để tính SVD
    max_k_y = max(k_y for k_y, _ in configs)
    max_k_c = max(k_c for _, k_c in configs)

    print("Computing SVD...")
    # 2. Tính SVD riêng cho kênh Y (max_k_y) và 2 kênh Cb, Cr (max_k_c)
    svd_y = SVD(
        channels[0], rank_tolerance=1e-3, max_iterations=250, n_components=max_k_y
    )
    svd_cb = SVD(
        channels[1], rank_tolerance=1e-3, max_iterations=250, n_components=max_k_c
    )
    svd_cr = SVD(
        channels[2], rank_tolerance=1e-3, max_iterations=250, n_components=max_k_c
    )
    svds = [svd_y, svd_cb, svd_cr]

    total_energy = CompressionMetrics.total_energy(Y)
    sigma_y = np.diagonal(svds[0].Sigma) if svds[0].Sigma.ndim == 2 else svds[0].Sigma

    table_data = []

    # 3. Duyệt qua từng cặp (k_y, k_c)
    for k_y, k_c in configs:
        # Reconstruct kênh Y với k_y, kênh Cb và Cr với k_c
        Y_k = reconstruct(svds[0], k_y)
        Cb_k = reconstruct(svds[1], k_c)
        Cr_k = reconstruct(svds[2], k_c)

        # Tính chỉ số nén bất đối xứng
        cr = CompressionMetrics.ycbcr_compression_ratio(k_y, k_c, m, n)
        energy = CompressionMetrics.energy_kept(sigma_y, k_y, total_energy)
        psnr = CompressionMetrics.psnr(Y, np.clip(Y_k, 0, 255))
        psnr_str = f"{psnr:.2f} dB" if np.isfinite(psnr) else "∞ dB"

        # Resize các kênh màu về kích thước gốc (n, m)
        channels_up = [
            np.array(
                Image.fromarray(np.clip(ch, 0, 255).astype(np.uint8)).resize(
                    (n, m), Image.Resampling.BILINEAR
                )
            )
            for ch in [Y_k, Cb_k, Cr_k]
        ]

        # Chuyển đổi YCbCr về RGB
        Y_f, Cb_f, Cr_f = channels_up
        Cb_f, Cr_f = Cb_f.astype(float) - 128, Cr_f.astype(float) - 128

        R = Y_f + 1.402 * Cr_f
        G = Y_f - 0.344136 * Cb_f - 0.714136 * Cr_f
        B = Y_f + 1.772 * Cb_f

        rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)

        # Đổi tên file lưu trữ để phân biệt Ky và Kc
        Image.fromarray(rgb).save(f"results/Ky{k_y}_Kc{k_c}.jpg")

        table_data.append(
            [f"Ky={k_y}, Kc={k_c}", f"{cr:.2f} : 1", f"{energy:.2f}%", psnr_str]
        )

    print(
        "\n"
        + tabulate(
            table_data,
            headers=[
                "Rank (k_Y, k_C)",
                "Compression Ratio",
                "Retained Energy (Y)",
                "PSNR (dB)",
            ],
            tablefmt="grid",
            numalign="right",
            stralign="center",
        )
    )


if __name__ == "__main__":
    run_ycbcr_compression_table()
