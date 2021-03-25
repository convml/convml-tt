from PIL import Image
import numpy as np


def create_true_color_img(das_channels):
    # https://github.com/blaylockbk/pyBKB_v2/blob/master/BB_GOES16/get_GOES16.py
    def contrast_correction(color, contrast):
        """
        Modify the contrast of an R, G, or B color channel
        See: #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
        Input:
            C - contrast level
        """
        F = (259 * (contrast + 255)) / (255.0 * 259 - contrast)
        COLOR = F * (color - 0.5) + 0.5
        COLOR = np.minimum(COLOR, 1)
        COLOR = np.maximum(COLOR, 0)
        return COLOR

    def channels_to_rgb(channels):
        B, R, G = channels

        # Turn empty values into nans
        R[R == -1] = np.nan
        G[G == -1] = np.nan
        B[B == -1] = np.nan

        # Apply range limits for each channel becuase RGB values must be between 0 and 1
        R = np.maximum(R, 0)
        R = np.minimum(R, 1)
        G = np.maximum(G, 0)
        G = np.minimum(G, 1)
        B = np.maximum(B, 0)
        B = np.minimum(B, 1)

        # Apply the gamma correction
        gamma = 0.4
        R = np.power(R, gamma)
        G = np.power(G, gamma)
        B = np.power(B, gamma)
        # print '\n   Gamma correction: %s' % gamma

        # Calculate the "True" Green
        G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
        G_true = np.maximum(G_true, 0)
        G_true = np.minimum(G_true, 1)

        # Modify the RGB color contrast:
        contrast = 80
        arr_img = np.dstack([R, G_true, B])
        return contrast_correction(arr_img, contrast)

    arr = channels_to_rgb([da.values / 256.0 for da in das_channels])
    img_data = (arr * 255).astype(np.uint8)
    return Image.fromarray(img_data)
