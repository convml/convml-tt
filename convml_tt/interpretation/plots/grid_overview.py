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
    axes=None,
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

    N_samples = len(idxs)
    if axes is not None:
        # make sure we turns lists into array
        axes = np.array(axes)
        if len(axes.flatten()) < N_samples:
            raise Exception(
                f"The axes you have provided aren't enough to fit the {N_samples} samples"
            )

        if len(axes.shape) == 2:
            nrows, ncols = axes.shape
        elif len(axes.shape) == 1:
            nrows, ncols = len(axes), 1
        else:
            raise NotImplementedError(axes.shape)

        fig = axes.flatten()[0].figure
    else:
        nrows = math.ceil(float(N_samples) / float(ncols))
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
            tile_image = tile_dataset.get_image(tile_id=i, tile_type=tile_type)
        elif isinstance(tile_dataset, ImageSingletDataset):
            tile_image = tile_dataset.get_image(tile_id=i)
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
            if "{" in label:
                label_text = label.format(**tile_dataset.df_tiles.loc[i].to_dict())
            else:
                label_text = tile_dataset.df_tiles.loc[i][label]

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
