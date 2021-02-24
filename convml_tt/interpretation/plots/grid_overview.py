import math

import matplotlib.pyplot as plt
import numpy as np

from ...architectures.triplet_trainer import TileType
from ...data.sources.imagelist import SingleTileImageList


def grid_overview(triplets_or_tilelist, points, tile=TileType.ANCHOR,
                  figwidth=16, ncols=10, label='tile_id'):
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

    nrows = math.ceil(float(len(idxs))/float(ncols))
    figheight = float(figwidth)/ncols*nrows
    figsize = (figwidth, figheight)

    if isinstance(triplets_or_tilelist, SingleTileImageList):
        if tile != TileType.ANCHOR:
            raise Exception("Only ANCHOR tiles are assumed to be in"
                            " SingleTileImageList objects, load triplets"
                            " if you need them")
        get_tile = lambda n: triplets_or_tilelist[n]
    else:
        get_tile = lambda n: triplets_or_tilelist[n][tile.value]

    lspace = 0.05
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             gridspec_kw=dict(hspace=lspace, wspace=lspace))

    for n, i in enumerate(idxs):
        ax = axes.flatten()[n]
        ax.axison = False
        tile_img = get_tile(i)
        tile_img.show(ax=ax)
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        label_text = None
        if type(label) == str and label == 'tile_id':
            label_text = str(i)
        elif type(label) == list:
            label_text = label[n]
        else:
            raise NotImplementedError(label)

        if label_text:
            ax.text(0.1, 0.1, label_text, transform=ax.transAxes,
                    bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 2})

    for n in range(len(idxs), nrows*ncols):
        ax = axes.flatten()[n]
        ax.axison = False
