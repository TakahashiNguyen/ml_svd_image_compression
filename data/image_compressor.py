import os

from PIL import Image

from .image_loader import ImageLoader
from .svd_processor import SVDProcessor
from .ycbcr_processor import YCbCrProcessor


class ImageCompressor:

    def __init__(self, image_path, output_dir="results"):
        self.image_path = image_path
        self.output_dir = output_dir

        self.svd_processor = SVDProcessor()
        self.ycbcr_processor = YCbCrProcessor()

        self.channels = None
        self.svds = None
        self.image_shape = None

    def load_image(self):
        """Đọc ảnh và chuyển sang không gian YCbCr."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(self.image_path)

        Y, Cb, Cr = ImageLoader().load_ycbcr(self.image_path)

        self.image_shape = Y.shape

        self.channels = self.ycbcr_processor.subsample_420(Y, Cb, Cr)

    def compute_svd(self, max_k_y, max_k_c):
        """
        Phân rã SVD độc lập cho ba kênh Y, Cb, Cr.
        """
        Y, Cb, Cr = self.channels

        self.svds = [
            self.svd_processor.decompose(Y, max_k_y),
            self.svd_processor.decompose(Cb, max_k_c),
            self.svd_processor.decompose(Cr, max_k_c),
        ]

    def compress(self, k_y, k_c):
        """
        Tái tạo từng kênh với rank tương ứng.
        """
        svd_y, svd_cb, svd_cr = self.svds

        Y_k = self.svd_processor.reconstruct(svd_y, k_y)

        Cb_k = self.svd_processor.reconstruct(svd_cb, k_c)

        Cr_k = self.svd_processor.reconstruct(svd_cr, k_c)

        m, n = self.image_shape

        rgb = self.ycbcr_processor.reconstruct_rgb(Y_k, Cb_k, Cr_k, (n, m))

        return rgb

    def save_image(self, rgb, k_y, k_c):
        """Lưu ảnh đã nén."""
        filename = f"Ky{k_y}_Kc{k_c}.jpg"

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, filename)

        Image.fromarray(rgb).save(output_path)

        return output_path
