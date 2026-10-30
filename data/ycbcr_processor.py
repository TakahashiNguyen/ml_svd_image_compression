import numpy as np
from PIL import Image


class YCbCrProcessor:
    """
    Xử lý chuyển đổi giữa không gian màu YCbCr và RGB.
    """

    @staticmethod
    def subsample_420(Y, Cb, Cr):
        """
        Thực hiện chroma subsampling 4:2:0.
        Giữ nguyên độ phân giải Y,
        giảm một nửa chiều cao và chiều rộng Cb, Cr.
        """
        return [Y, Cb[::2, ::2], Cr[::2, ::2]]

    @staticmethod
    def resize_channel(channel, size):
        """Phóng to kênh màu về kích thước ban đầu."""
        channel = np.clip(channel, 0, 255)
        channel = channel.astype(np.uint8)

        return np.array(
            Image.fromarray(channel).resize(size, Image.Resampling.BILINEAR)
        ).astype(float)

    @staticmethod
    def to_rgb(Y, Cb, Cr):
        """
        Chuyển đổi YCbCr sang RGB.
        """
        Cb = Cb.astype(float) - 128
        Cr = Cr.astype(float) - 128

        R = Y + 1.402 * Cr

        G = Y - 0.344136 * Cb - 0.714136 * Cr

        B = Y + 1.772 * Cb

        rgb = np.stack([R, G, B], axis=-1)

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def reconstruct_rgb(self, Y, Cb, Cr, original_size):
        """
        Phóng to các kênh màu và khôi phục ảnh RGB.
        """
        Y = self.resize_channel(Y, original_size)
        Cb = self.resize_channel(Cb, original_size)
        Cr = self.resize_channel(Cr, original_size)

        return self.to_rgb(Y, Cb, Cr)
