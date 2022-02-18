import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ...data.dataset import ImageSingletDataset, ImageTripletDataset, TileType


def grid_overview(
    tile_dataset: Union[ImageSingletDataset, ImageTripletDataset, xr.DataArray],
    points,
    tile_type: TileType = None,
    figwidth=16,
    ncols=10,
    label="tile_id",
):
    """
    Plot a grid overview of the chosen tile at the selected `points`. If
    `points` is an integer the first N=points tiles will be plotted, if
    `points` is a list it will be interpreted as indecies into `triplets`
    """
    if isinstance(points, int):
        idxs = np.arange(points)
    elif isinstance(points, np.ndarray):
        idxs = points
    elif isinstance(points, list):
        idxs = np.array(points)
    else:
        raise NotImplementedError(type(points))

    nrows = math.ceil(float(len(idxs)) / float(ncols))
    figheight = float(figwidth) / ncols * nrows
    figsize = (figwidth, figheight)

    lspace = 0.05
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=dict(hspace=lspace, wspace=lspace),
    )

    for n, i in enumerate(idxs):
        ax = axes.flatten()[n]
        ax.axison = False
        if isinstance(tile_dataset, ImageTripletDataset):
            tile_image = tile_dataset.get_image(index=i, tile_type=tile_type)
        elif isinstance(tile_dataset, ImageSingletDataset):
            tile_image = tile_dataset.get_image(index=i)
        else:
            raise NotImplementedError(tile_dataset)

        ax.imshow(tile_image)
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        label_text = None
        if type(label) == str and label == "tile_id":
            label_text = str(i)
        elif type(label) == list:
            label_text = label[n]
        else:
            raise NotImplementedError(label)

        if label_text:
            ax.text(
                0.1,
                0.1,
                label_text,
                transform=ax.transAxes,
                bbox={"facecolor": "white", "alpha": 0.4, "pad": 2},
            )

    for n in range(len(idxs), nrows * ncols):
        ax = axes.flatten()[n]
        ax.axison = False
