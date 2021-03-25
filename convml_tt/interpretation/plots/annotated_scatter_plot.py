from collections import OrderedDict
import enum

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from ...data.dataset import ImageSingletDataset
from .mpl_autopos_annotation import calc_offset_points
from .mpl_autopos_annotation.convex_hull import (
    calc_point_offsets as calc_offset_points_ch,
)


def find_nearest_tile(x_sample, y_sample, x, y, dim="tile_id", scaling=1.0):
    """
    Given the data arrays `x` and `y` which have the dimension `tile_id`, i.e.
    there's a value for each tile, find the nearest `tile_id`s to each point
    pair in `x_sample`, `y_sample`. Optionally `scaling` can be provided in
    case the range of values in x and y are very different
    """
    # x_sample and y_sample might just be numpy arrays so create data arrays
    # here so we get the correct broadcasting
    x_sample = xr.DataArray(x_sample, dims=("point",))
    y_sample = xr.DataArray(y_sample, dims=("point",))

    dx = x_sample - x
    dy = (y_sample - y) * scaling

    dl = np.sqrt(dx * dx + dy * dy)

    # can't simply use `argmin` here on `dl` since that would return the index
    # in dl, not the `tile_id`
    return dl.isel(**{dim: dl.argmin(dim=dim)})[dim]


def annotated_scatter_plot(
    x,
    y,
    points,
    ax=None,
    size=0.1,
    autopos_method="forces",
    use_closest_point=True,
    annotation_dist=1.0,
    hue=None,
    hue_palette="hls",
    tile_dataset: ImageSingletDataset = None,
):
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

    required_vars = ["data_dir", "tile_type", "stage"]
    if tile_dataset is None:
        kws = None
        if "data_dir" in x.attrs:
            kws = {v: x.attrs[v] for v in required_vars}
        elif "data_dir" in y.attrs:
            kws = {v: y.attrs[v] for v in required_vars}

        if kws is not None:
            tile_dataset = ImageSingletDataset(**kws)

    if tile_dataset is None:
        raise Exception(
            f"Couldn't find the required values {required_vars} in attr of "
            "`x` or `y`. Either set these or provide a `tile_dataset`"
        )

    def _is_array(v):
        return isinstance(v, np.ndarray) or (type(v) == list and type(v[0]) == int)

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
        xl = x.max() - x.min()
        yl = y.max() - y.min()
        s = xl / yl
        tile_ids = find_nearest_tile(x_sample=x_c, y_sample=y_c, x=x, y=y, scaling=s)

        # For the annotations we want to draw a line to where these tiles
        # actually exist
        x_sample = x.sel(tile_id=tile_ids)
        y_sample = y.sel(tile_id=tile_ids)
    else:
        raise NotImplementedError(type(points))

    pts = np.array([x_sample, y_sample]).T
    # if tiles are spaced uniformly round a circle, then
    # N*s=2*pi*r
    # => r ~ N*s/6.
    # and we want the radius to be three units relative to the size, so
    scale = pts.shape[0] * size / 2.0 * annotation_dist

    if autopos_method == "forces":
        pts_offset = calc_offset_points(pts, scale=scale)
    elif autopos_method == "convex_hull":
        pts_offset = calc_offset_points_ch(pts, scale=scale)
    else:
        raise NotImplementedError(autopos_method)

    # we plot points that will be hidden by the tile images because this helps
    # with autoscaling the axes
    ax.scatter(*pts_offset.T, alpha=0.0)
    ax.margins(0.2)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def transform(coord):
        return (ax.transData + fig.transFigure.inverted()).transform(coord)

    lines = []

    if hue is not None:
        if not isinstance(hue, xr.DataArray) or "tile_id" not in hue.dims:
            raise Exception(
                "`hue` should be a data-array with a label for"
                " for each tile in `x` and `y` (i.e. have the "
                " the same `tile_id`s)"
            )

        color_palette_name = hue_palette
        hue_unique = np.sort(np.unique(hue))
        color_palette = sns.color_palette(color_palette_name, n_colors=len(hue_unique))
        colormap = OrderedDict(zip(hue_unique, color_palette))
    else:
        color = "grey"

    for n, tile_id in enumerate(tile_ids):
        x_, y_ = pts_offset[n]

        pts_connector = np.c_[pts_offset[n], pts[n]]
        if hue is not None:
            color = colormap[hue.sel(tile_id=tile_id).item()]
        (line,) = ax.plot(*pts_connector, linestyle="--", marker=".", color=color)

        if x_c is not None and y_c is not None:
            if x_err is not None and y_err is not None:
                ax.errorbar(
                    x=x_c[n],
                    y=y_c[n],
                    xerr=x_err[n],
                    yerr=y_err[n],
                    marker="+",
                    color=line.get_color(),
                    capsize=2,
                )
            else:
                ax.scatter(x=x_c[n], y=y_c[n], marker="+", color=line.get_color())

        # sc = ax.scatter(*pts[n], marker='.', color=line.get_color())
        lines.append(line)

        xp, yh = transform((x_, y_))

        ax1 = fig.add_axes([xp - 0.5 * size, yh - size * 0.5, size, size])
        ax1.set_aspect(1)
        ax1.axison = False

        if hue is not None:
            label_text = hue.sel(tile_id=tile_id).item()
            if isinstance(label_text, xr.DataArray):
                label_text = label_text.values
            ax1.text(
                0.1,
                0.15,
                label_text,
                transform=ax1.transAxes,
                bbox={"facecolor": "white", "alpha": 0.6, "pad": 2},
            )

        img_idx = int(tile_id.values)
        img = tile_dataset.get_image(index=img_idx)

        # img = Image.open(tiles_path/"{:05d}_anchor.png".format())
        ax1.imshow(img)

    # ensure that adding the points doesn't change the axis limits since we
    # calculated the offset for the annotations
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if hue is not None:
        sns.scatterplot(
            x,
            y,
            hue=hue.values,
            ax=ax,
            alpha=0.4,
            palette=color_palette,
            hue_order=colormap.keys(),
        )
    else:
        ax.scatter(x, y, marker=".", alpha=0.2, color="grey")

    ax.set_xlabel(x._title_for_slice() + "\n" + xr.plot.utils.label_from_attrs(x))
    ax.set_ylabel(y._title_for_slice() + "\n" + xr.plot.utils.label_from_attrs(y))

    return lines, pts
