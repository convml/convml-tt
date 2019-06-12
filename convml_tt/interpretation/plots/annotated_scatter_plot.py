import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as il
import xarray as xr

from ...utils import get_triplets_from_encodings

from .mpl_autopos_annotation import calc_offset_points
from .mpl_autopos_annotation.convex_hull import calc_point_offsets as calc_offset_points_ch

from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline


def find_nearest_tile(x_sample, y_sample, x, y, dim='tile_id', scaling=1.0):
    """
    Given the data arrays `x` and `y` which have the dimension `tile_id`, i.e.
    there's a value for each tile, find the nearest `tile_id`s to each point
    pair in `x_sample`, `y_sample`. Optionally `scaling` can be provided in
    case the range of values in x and y are very different
    """
    # x_sample and y_sample might just be numpy arrays so create data arrays
    # here so we get the correct broadcasting
    x_sample = xr.DataArray(x_sample, dims=('point',))
    y_sample = xr.DataArray(y_sample, dims=('point',))

    dx = x_sample - x
    dy = (y_sample - y)*scaling

    dl = np.sqrt(dx*dx + dy*dy)

    return dl.argmin(dim=dim)


def scatter_annotated(x, y, points, ax=None, size=0.1, autopos_method='forces',
                      use_closest_point=True):
    """
    create scatter plot from values in `x` and `y` picking out points to
    highlight with a tile-graphic annotation based on what is passed in as
    `points`

    if `points` is
        - of type `int`:
            number of points to randomly pick from `x` and `y`
        - a list or `np.array` of `ints`:
            indecies to index into `x` and `y`
        - tuple containing two `np.arrays`:
            actual points to plot, the nearest (x,y)-point will be used for
            annotation. The points provided will be plotted with `+` markers
        - tuple containing two tuples of `np.arrays`:
            the outer tuples will be assumed to be representing `x` and `y`
            values, with the first element of the tuples being actual points to
            plot and the second the errorbars to plot. As above the nearest
            tiles to the point will be used for annotation and the points
            provided will be plotted with `+` markers
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = ax.figure

    triplets = get_triplets_from_encodings(x)

    _is_array = lambda v: isinstance(v, np.ndarray)

    x_err, y_err, x_c, y_c = None, None, None, None
    if type(points) == int:
        N = len(x)
        idx_sample = np.random.choice(np.arange(N), size=points)
        x_sample, y_sample = x[idx_sample], y[idx_sample]
        tile_ids = x_sample.tile_id
    elif _is_array(points):
        x_sample = x.sel(tile_id=points)
        y_sample = y.sel(tile_id=points)
        tile_ids = x_sample.tile_id
    elif isinstance(points, tuple) and len(points) == 2:
        try:
            if _is_array(points[0]) and _is_array(points[1]):
                x_c, y_c = points
            else:
                (x_c, x_err), (y_c, y_err) = points
        except Exception as e:
            raise NotImplementedError(type(points), e)
        tile_ids = find_nearest_tile(x_sample=x_c, y_sample=y_c, x=x, y=y)

        # For the annotations we want to draw a line to where these tiles
        # actually exist
        x_sample = x.sel(tile_id=tile_ids)
        y_sample = y.sel(tile_id=tile_ids)
    else:
        raise NotImplementedError(type(points))

    ax.scatter(x, y, marker='.', alpha=0.2, color='grey')

    ax.set_xlabel(x._title_for_slice() + '\n' + xr.plot.utils.label_from_attrs(x))
    ax.set_ylabel(y._title_for_slice() + '\n' + xr.plot.utils.label_from_attrs(y))

    pts = np.array([x_sample, y_sample]).T
    if autopos_method == 'forces':
        pts_offset = calc_offset_points(pts, scale=3*size)
    elif autopos_method == 'convex_hull':
        pts_offset = calc_offset_points_ch(pts, scale=3*size)
    else:
        raise NotImplementedError(autopos_method)

    # we plot points that will be hidden by the tile images because this helps
    # with autoscaling the axes
    ax.scatter(*pts_offset.T, alpha=0.0)
    ax.margins(0.2)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def transform(coord):
        return (ax.transData + fig.transFigure.inverted()).transform(coord)

    for n, tile_id in enumerate(tile_ids):
        x_, y_ = pts_offset[n]

        pts_connector = np.c_[pts_offset[n], pts[n]]
        line, = ax.plot(*pts_connector, linestyle='--', alpha=0.5)

        if x_c is not None and y_c is not None:
            if x_err is not None and y_err is not None:
                ax.errorbar(x=x_c[n], y=y_c[n], xerr=x_err[n], yerr=y_err[n],
                            marker='+', color=line.get_color(), capsize=2)
            else:
                ax.scatter(x=x_c[n], y=y_c[n], marker='+',
                           color=line.get_color())

        sc = ax.scatter(*pts[n], marker='.', color=line.get_color())

        xp, yh = transform((x_, y_))

        ax1=fig.add_axes([xp-0.5*size, yh-size*0.5, size, size])
        ax1.set_aspect(1)
        ax1.axison = False

        img_idx = int(tile_id.values)
        img = triplets[img_idx]

        # img = Image.open(tiles_path/"{:05d}_anchor.png".format())
        ax1.imshow(img)

    # ensure that adding the points doesn't change the axis limits since we
    # calculated the offset for the annotations
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return sc, pts
