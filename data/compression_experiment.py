import numpy as np
from tabulate import tabulate

from data.metrics import CompressionMetrics

from .image_compressor import ImageCompressor


class CompressionExperiment:

    def __init__(self, image_path, configs):
        self.image_path = image_path
        self.configs = configs

        self.compressor = ImageCompressor(image_path)

        self.results = []

    def prepare(self):
        """Đọc ảnh và thực hiện SVD một lần."""
        self.compressor.load_image()

        max_k_y = max(k_y for k_y, _ in self.configs)
        max_k_c = max(k_c for _, k_c in self.configs)

        self.compressor.compute_svd(max_k_y, max_k_c)

    def evaluate(self, k_y, k_c):
        """Đánh giá một cấu hình nén."""

        rgb = self.compressor.compress(k_y, k_c)

        output_path = self.compressor.save_image(rgb, k_y, k_c)

        Y = self.compressor.channels[0]

        svd_y = self.compressor.svds[0]

        sigma_y = self.compressor.svd_processor.get_singular_values(svd_y)

        m, n = self.compressor.image_shape

        total_energy = CompressionMetrics.total_energy(Y)

        energy = CompressionMetrics.energy_kept(sigma_y, k_y, total_energy)

        Y_k = self.compressor.svd_processor.reconstruct(svd_y, k_y)

        psnr = CompressionMetrics.psnr(Y, np.clip(Y_k, 0, 255))

        cr = CompressionMetrics.ycbcr_compression_ratio(k_y, k_c, m, n)

        psnr_str = f"{psnr:.2f} dB" if np.isfinite(psnr) else "∞ dB"

        # Frobenius reconstruction error
        frobenius_error = CompressionMetrics.frobenius_error(Y, Y_k)

        result = {
            "rank": f"Ky={k_y}, Kc={k_c}",
            "cr": f"{cr:.2f} : 1",
            "frobenius_error": f"{frobenius_error:.4f}",
            "energy": f"{energy:.2f}%",
            "psnr": psnr_str,
            "path": output_path,
        }

        self.results.append(result)

    def run(self):
        """Chạy toàn bộ thực nghiệm."""
        self.prepare()

        for k_y, k_c in self.configs:
            self.evaluate(k_y, k_c)

        self.display_results()

    def display_results(self):
        """Hiển thị bảng đánh giá."""

        table_data = [
            [r["rank"], r["cr"], r["frobenius_error"], r["energy"], r["psnr"]]
            for r in self.results
        ]

        print(
            "\n"
            + tabulate(
                table_data,
                headers=[
                    "Rank (k_Y, k_C)",
                    "Compression Ratio",
                    "Frobenius Error (Y)",
                    "Retained Energy (Y)",
                    "PSNR (dB)",
                ],
                tablefmt="grid",
                numalign="right",
                stralign="center",
            )
        )
