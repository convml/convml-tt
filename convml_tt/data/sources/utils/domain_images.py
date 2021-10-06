"""
Utilities for creating a images representing the scene source data
"""
import numpy as np
from PIL import Image

from .. import goes16


def rgb_image_from_scene_data(data_source, da_scene, src_attrs):
    if data_source.source == "goes16" and data_source.type == "truecolor_rgb":
        # before creating the image with satpy we need to set the attrs
        # again to ensure we get a proper RGB image
        da_scene.attrs.update(src_attrs)
        img_domain = goes16.satpy_rgb.rgb_da_to_img(da=da_scene)
    else:
        # TODO: make a more generic image generation function
        img_data = da_scene.data
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        img_data = (img_data * 255).astype(np.uint8)
        img_domain = Image.fromarray(img_data)

    return img_domain
