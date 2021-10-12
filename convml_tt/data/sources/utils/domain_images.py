"""
Utilities for creating a images representing the scene source data
"""
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

from .. import goes16


def save_ax_nosave(ax, **kwargs):
    """
    Create an Image object from the content of a matplotlib.Axes which can be saved to a PNG-file

    source: https://stackoverflow.com/a/43099136/271776
    """
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    ax.figure.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox, **kwargs)
    ax.axis("on")
    buff.seek(0)
    return Image.open(buff)


def rgb_image_from_scene_data(data_source, da_scene, src_attrs):
    if data_source.source == "goes16":
        if data_source.type == "truecolor_rgb" and da_scene.name == "Rad":
            # before creating the image with satpy we need to set the attrs
            # again to ensure we get a proper RGB image
            da_scene.attrs.update(src_attrs)
            img_domain = goes16.satpy_rgb.rgb_da_to_img(da=da_scene)
        else:
            height = 5.0
            lx, ly = (
                da_scene.x.max() - da_scene.x.min(),
                da_scene.y.max() - da_scene.y.min(),
            )
            width = height / ly * lx
            fig, ax = plt.subplots(figsize=(width, height))
            ax.set_aspect(1.0)
            da_scene.plot(ax=ax, cmap="nipy_spectral", y="y", add_colorbar=False)
            img_domain = save_ax_nosave(ax=ax)
    else:
        # TODO: make a more generic image generation function
        img_data = da_scene.data
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        img_data = (img_data * 255).astype(np.uint8)
        img_domain = Image.fromarray(img_data)

    return img_domain


def align_axis_x(ax, ax_target):
    """Make x-axis of `ax` aligned with `ax_target` in figure"""
    posn_old, posn_target = ax.get_position(), ax_target.get_position()
    ax.set_position([posn_target.x0, posn_old.y0, posn_target.width, posn_old.height])
