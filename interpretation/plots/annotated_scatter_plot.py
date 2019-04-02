import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as il

from ...utils import get_triplets_from_encodings

from .utils import calc_point_offsets

from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline

def scatter_annotated(x, y, points, ax=None, size=0.2):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = ax.figure

    triplets = get_triplets_from_encodings(x)

    if type(points) == int:
        N = len(x)
        idx_sample = np.random.choice(np.arange(N), size=points)
        x_sample, y_sample = x[idx_sample], y[idx_sample]
    elif type(points) == np.ndarray:
        x_sample = x.sel(tile_id=points)
        y_sample = y.sel(tile_id=points)
    else:
        raise NotImplementedError(type(points))
        
    ax.scatter(x, y, marker='.', alpha=0.2, color='grey')

    ax.scatter(x_sample, y_sample, marker='.')

    ax.set_xlabel(x._title_for_slice())
    ax.set_ylabel(y._title_for_slice())

    pts = np.array([x_sample, y_sample]).T
    pts_offset = calc_point_offsets(pts, scale=3*size)

    def transform(coord):
        return (ax.transData + fig.transFigure.inverted()).transform(coord)

    for n, tile_id in enumerate(x_sample.tile_id):
        x_, y_ = pts_offset[n]
        #tile_id, color = tile_ids[n]

        pts_connector = np.c_[pts_offset[n], pts[n]]
        ax.plot(*pts_connector, linestyle='--', alpha=0.5)

        xp, yh = transform((x_, y_))

        ax1=fig.add_axes([xp-0.5*size, yh-size*0.5, size, size])
        ax1.set_aspect(1)
        ax1.axison = False

        img_idx = int(tile_id.values)
        img = triplets[img_idx]

        # img = Image.open(tiles_path/"{:05d}_anchor.png".format())
        ax1.imshow(img)
