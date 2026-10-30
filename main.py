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

    max_k = 640
    configs = [max_k / 5, max_k / 3, max_k * 2 / 3, max_k]

    # Subsample chroma channels (4:2:0)
    channels = [Y, Cb[::2, ::2], Cr[::2, ::2]]

    print("Computing SVD...")
    svds = [
        SVD(channel, rank_tolerance=1e-3, max_iterations=250, n_components=max_k)
        for channel in channels
    ]

    total_energy = CompressionMetrics.total_energy(Y)
    sigma_y = np.diagonal(svds[0].Sigma) if svds[0].Sigma.ndim == 2 else svds[0].Sigma

    table_data = []

    for k in configs:
        k = int(k)

        # Reconstruct three channels
        Y_k, Cb_k, Cr_k = [reconstruct(svd, k) for svd in svds]

        # Calculate metrics
        cr = CompressionMetrics.ycbcr_compression_ratio(k, k, m, n)
        energy = CompressionMetrics.energy_kept(sigma_y, k, total_energy)
        psnr = CompressionMetrics.psnr(Y, np.clip(Y_k, 0, 255))
        psnr_str = f"{psnr:.2f} dB" if np.isfinite(psnr) else "∞ dB"

        # Resize channels to original dimensions
        channels_up = [
            np.array(
                Image.fromarray(np.clip(ch, 0, 255).astype(np.uint8)).resize(
                    (n, m), Image.Resampling.BILINEAR
                )
            )
            for ch in [Y_k, Cb_k, Cr_k]
        ]

        # Convert YCbCr to RGB
        Y_f, Cb_f, Cr_f = channels_up
        Cb_f, Cr_f = Cb_f.astype(float) - 128, Cr_f.astype(float) - 128

        R = Y_f + 1.402 * Cr_f
        G = Y_f - 0.344136 * Cb_f - 0.714136 * Cr_f
        B = Y_f + 1.772 * Cb_f

        rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(f"results/{k}.jpg")

        table_data.append([k, f"{cr:.2f} : 1", f"{energy:.2f}%", psnr_str])

    print(
        "\n"
        + tabulate(
            table_data,
            headers=["rank-K", "Compression Ratio", "Retained Energy (Y)", "PSNR (dB)"],
            tablefmt="grid",
            numalign="right",
            stralign="center",
        )
    )

    print("\nĐã hoàn tất nén ảnh. Kết quả được lưu trong thư mục results.")


if __name__ == "__main__":
    run_ycbcr_compression_table()
