import numpy as np
from PIL import Image


class ImageLoader:
    """Load an image file and convert to YCbCr color space using Pillow."""

    def load_ycbcr(self, path):
        """Load an image file and return three matrices: Y, Cb, Cr.

        Parameters
        ----------
        path : str
            Path to the image file.

        Returns
        -------
        Y : ndarray
            Luminance matrix (0..255)
        Cb : ndarray
            Blue-difference chroma matrix (0..255)
        Cr : ndarray
            Red-difference chroma matrix (0..255)
        """
        img = Image.open(path).convert("YCbCr")
        ycbcr_array = np.array(img, dtype=np.uint8)

        Y = ycbcr_array[:, :, 0]
        Cb = ycbcr_array[:, :, 1]
        Cr = ycbcr_array[:, :, 2]

        return Y, Cb, Cr
