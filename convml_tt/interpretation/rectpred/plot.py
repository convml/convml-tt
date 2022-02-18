import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import xarray as xr


def make_rgb(da, alpha=0.5, **coord_components):
    """
    turn three components along a particular coordinate into RGB values by
    scaling each by its max and min-values in the dataset

    >> make_rgb(da, emb_dim=[0,1,2])
    """
    if len(coord_components) != 1:
        raise Exception(
            "You should provide exactly one coordinate to turn into RGB values"
        )

    v_dim, dim_idxs = list(coord_components.items())[0]

    if len(dim_idxs) != 3:
        raise Exception(
            f"You should provide exactly three indexes of the `{v_dim}` coordinate to turn into RGB"
        )
    elif v_dim not in da.dims:
        raise Exception(f"The `{v_dim}` coordinate wasn't found the provided DataArray")

    def scale_zero_one(v):
        return (v - v.min()) / (v.max() - v.min())

    scale = scale_zero_one

    all_dims = da.dims
    x_dim, y_dim = list(filter(lambda d: d != v_dim, all_dims))

    da_rgba = xr.DataArray(
        np.zeros((4, len(da[x_dim]), len(da[y_dim]))),
        dims=("rgba", x_dim, y_dim),
        coords={"rgba": np.arange(4), x_dim: da[x_dim], y_dim: da[y_dim]},
    )

    def _make_component(da_):
        if da_.rgba.data == 3:
            return alpha * np.ones_like(da_)
        else:
            return scale(da.sel({v_dim: dim_idxs[da_.rgba.item()]}).values)

    da_rgba = da_rgba.groupby("rgba").apply(_make_component)

    return da_rgba


def get_img_with_extent_cropped(da_emb):
    """
    Load the image in `img_fn`, clip the image and return the
    image extent (xy-extent if the source data coordinates are available)
    """
    img = _load_image(da_emb=da_emb)

    i_ = da_emb.i0.values
    # NB: j-values might be decreasing in number if we're using a
    # y-coordinate with increasing values from bottom left, so we
    # sort first
    j_ = np.sort(da_emb.j0.values)

    def get_spacing(v):
        dv_all = np.diff(v)
        assert np.all(dv_all[0] == dv_all)
        return dv_all[0]

    i_step = get_spacing(i_)
    j_step = get_spacing(j_)

    ilim = (i_.min() - i_step // 2, i_.max() + i_step // 2)
    jlim = (j_.min() - j_step // 2, j_.max() + j_step // 2)

    img = img[slice(*jlim), slice(*ilim)]

    if "x" in da_emb.coords and "y" in da_emb.coords:
        # get x,y and indexing (i,j) extent from data array so
        # so we can clip and plot the image correctly
        x_min = da_emb.x.min()
        y_min = da_emb.y.min()
        x_max = da_emb.x.max()
        y_max = da_emb.y.max()

        dx = get_spacing(da_emb.x.values)
        dy = get_spacing(da_emb.y.values)
        xlim = (x_min - dx // 2, x_max + dx // 2)
        ylim = (y_min - dy // 2, y_max + dy // 2)

        extent = [*xlim, *ylim]

    return img, np.array(extent)


def get_img_with_extent(da_emb):
    """
    Load the image in `img_fn` and return the
    image extent (xy-extent if the source data coordinates are available)
    """
    img = _load_image(da_emb=da_emb)
    img_extent = _load_image_extent(da_emb=da_emb)
    return img, img_extent


def plot_scene_image(da_emb, crop_image=True, ax=None):
    """
    Render the RGB image of the scene with `scene_id` in `da` into `ax` (a new
    figure will be created if `ax=None`)
    """
    if len(da_emb.shape) == 3:
        # ensure non-xy dim is first
        d_not_xy = list(filter(lambda d: d not in ["x", "y"], da_emb.dims))
        da_emb = da_emb.transpose(*d_not_xy, "x", "y")

    img = _load_image(da_emb=da_emb)
    img_extent = _load_image_extent(da_emb=da_emb)

    if ax is None:
        lx = img_extent[1] - img_extent[0]
        ly = img_extent[3] - img_extent[2]
        r = lx / ly
        fig_height = 3.0
        fig_width = fig_height * r
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect(1.0)
        ax.set_xlabel(xr.plot.utils.label_from_attrs(da_emb.x))
        ax.set_ylabel(xr.plot.utils.label_from_attrs(da_emb.y))
    ax.imshow(img, extent=img_extent, rasterized=True)

    return ax, img_extent, img


def _load_image(da_emb):
    if "image_path" not in da_emb.attrs:
        raise Exception(
            "Cannot plot embedding annotation map images without the `image_path` "
            "attribute defined for the embedding data-array (as this defines the "
            "path of the image to overlay onto)"
        )

    return mpimg.imread(da_emb.image_path)


def _load_image_extent(da_emb):
    if "src_data_path" not in da_emb.attrs:
        raise Exception(
            "Cannot plot embedding annotation map images without the `src_data_path` "
            "attribute defined for the embedding data-array (as this defines the "
            "extent of the image to overlay onto)"
        )

    da_src = xr.open_dataarray(da_emb.src_data_path)
    return [da_src.x.min(), da_src.x.max(), da_src.y.min(), da_src.y.max()]


def make_rgb_annotation_map_image(
    da_emb, rgb_components, render_tiles=False, crop_image=True
):
    """
    Render the contents of `da` onto the RGB image represented by the scene
    (`scene_id` expected to be defined for `da`).

    If `da` is 2D it will be rendered with discrete colours, otherwise if `da`
    is 3D the components of da to use should be given by `rgb_components`
    """
    if len(da_emb.shape) == 3:
        # ensure non-xy dim is first
        d_not_xy = list(filter(lambda d: d not in ["x", "y"], da_emb.dims))
        da_emb = da_emb.transpose(*d_not_xy, "x", "y")

    # now we get to adding the annotation

    # scale distances to km
    if da_emb.x.units == "m" and da_emb.y.units == "m":
        s = 1000.0
        da_emb = da_emb.copy().assign_coords(
            dict(x=da_emb.x.values / s, y=da_emb.y.values / s)
        )
        da_emb.x.attrs["units"] = "km"
        da_emb.y.attrs["units"] = "km"
    else:
        s = 1.0

    if len(da_emb.shape) == 3:
        da_rgba = make_rgb(da=da_emb, alpha=0.5, **{d_not_xy[0]: rgb_components})
    elif len(da_emb.shape) == 2:
        # when we have distinct classes (identified by integers) we just
        # want to map each label to a RGB color
        labels = da_emb.stack(dict(n=da_emb.dims))
        arr_rgb = skimage.color.label2rgb(label=labels.values, bg_color=(1.0, 1.0, 1.0))
        # make an RGBA array so we can apply some alpha blending later
        rgba_shape = list(arr_rgb.shape)
        rgba_shape[-1] += 1
        arr_rgba = 0.3 * np.ones((rgba_shape))
        arr_rgba[..., :3] = arr_rgb
        # and put this into a DataArray, unstack to recover original dimensions
        da_rgba = xr.DataArray(
            arr_rgba, dims=("n", "rgba"), coords=dict(n=labels.n)
        ).unstack("n")
    else:
        raise NotImplementedError(da_emb.shape)

    # set up the figure
    nrows = render_tiles and 4 or 3
    fig, axes = plt.subplots(
        figsize=(8, 3.2 * nrows),
        nrows=nrows,
        subplot_kw=dict(aspect=1),
        sharex=True,
    )

    ax = axes[0]
    _, img_extent, img = plot_scene_image(da_emb=da_emb, ax=ax, crop_image=crop_image)

    ax = axes[1]
    plot_scene_image(da_emb=da_emb, ax=ax, crop_image=crop_image)
    da_rgba.plot.imshow(ax=ax, rgb="rgba", y="y", rasterized=True)

    ax = axes[2]
    da_rgba[3] = 1.0
    da_rgba.plot.imshow(ax=ax, rgb="rgba", y="y", rasterized=True)

    if "lx_tile" in da_emb.attrs and "ly_tile" in da_emb.attrs:
        if render_tiles:
            x_, y_ = xr.broadcast(da_emb.x, da_emb.y)
            axes[2].scatter(x_, y_, marker="x")

            lx = da_emb.lx_tile / s
            ly = da_emb.ly_tile / s
            ax = axes[3]
            ax.imshow(img, extent=img_extent)
            for xc, yc in zip(x_.values.flatten(), y_.values.flatten()):
                c = da_rgba.sel(x=xc, y=yc).astype(float)
                # set alpha so we can see overlapping tiles
                c[-1] = 0.2
                c_edge = np.array(c)
                c_edge[-1] = 0.5
                rect = mpatches.Rectangle(
                    (xc - lx / 2.0, yc - ly / 2),
                    lx,
                    ly,
                    linewidth=1,
                    edgecolor=c_edge,
                    facecolor=c,
                    linestyle=":",
                )
                ax.scatter(xc, yc, color=c[:-1], marker="x")

                ax.add_patch(rect)

            def pad_lims(lim):
                plen = min(lim[1] - lim[0], lim[3] - lim[2])
                return [
                    (lim[0] - 0.1 * plen, lim[1] + plen * 0.1),
                    (lim[2] - 0.1 * plen, lim[3] + plen * 0.1),
                ]

            xlim, ylim = pad_lims(img_extent)
        else:
            x0, y0 = da_emb.x.min(), da_emb.y.max()
            lx = da_emb.lx_tile / s
            ly = da_emb.ly_tile / s
            rect = mpatches.Rectangle(
                (x0 - lx / 2.0, y0 - ly / 2),
                lx,
                ly,
                linewidth=1,
                edgecolor="grey",
                facecolor="none",
                linestyle=":",
            )
            ax.add_patch(rect)

    xlim = img_extent[:2]
    ylim = img_extent[2:]

    [ax.set_xlim(xlim) for ax in axes]
    [ax.set_ylim(ylim) for ax in axes]
    [ax.set_aspect(1) for ax in axes]

    plt.tight_layout()

    return fig, axes


def make_components_annotation_map_image(da_emb, components=[0, 1, 2], col_wrap=2):
    """
    Create overlay plots of individual embedding components as individual
    subplots each overlaid on the scene image
    """
    da_emb.coords["pca_dim"] = np.arange(da_emb.pca_dim.count())

    da_emb = da_emb.assign_coords(
        x=da_emb.x / 1000.0,
        y=da_emb.y / 1000.0,
        explained_variance=np.round(da_emb.explained_variance, 2),
    )
    da_emb.x.attrs["units"] = "km"
    da_emb.y.attrs["units"] = "km"

    img = _load_image(da_emb=da_emb)
    img_extent = _load_image_extent(da_emb=da_emb)

    img_extent = np.array(img_extent) / 1000.0

    # find non-xy dim
    d_not_xy = next(filter(lambda d: d not in ["x", "y"], da_emb.dims))

    N_subplots = len(components) + 1
    data_r = 3.0
    ncols = col_wrap
    size = 3.0

    nrows = int(np.ceil(N_subplots / ncols))
    figsize = (int(size * data_r * ncols), int(size * nrows))

    fig, axes = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        subplot_kw=dict(aspect=1),
        sharex=True,
    )

    ax = axes.flatten()[0]
    ax.imshow(img, extent=img_extent)
    ax.set_title(da_emb.scene_id.item())

    for n, ax in zip(components, axes.flatten()[1:]):
        ax.imshow(img, extent=img_extent)
        da_ = da_emb.sel(**{d_not_xy: n})
        da_ = da_.drop(["i0", "j0", "scene_id"])

        da_.plot.imshow(ax=ax, y="y", alpha=0.5, add_colorbar=False)

        ax.set_xlim(*img_extent[:2])
        ax.set_ylim(*img_extent[2:])

    [ax.set_aspect(1) for ax in axes.flatten()]
    [ax.set_xlabel("") for ax in axes[:-1, :].flatten()]

    fig.tight_layout()

    fig.text(
        0.0,
        -0.02,
        "cum. explained variance: {}".format(
            np.cumsum(da_emb.explained_variance.values)
        ),
    )
    return fig, axes
