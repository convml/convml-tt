#!/usr/bin/env python
# coding: utf-8

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.distributions.empirical_distribution as edf
import xarray as xr
from PIL import Image
from scipy.interpolate import interp1d
from tqdm import tqdm

from ...data.common import TRIPLET_TILE_IDENTIFIER_FORMAT
from .. import embedding_transforms
from . import annotated_scatter_plot


def _vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(
        np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
    )


def interp_ecfd(sample):
    # https://stackoverflow.com/a/44163082
    sample_edf = edf.ECDF(sample)
    slope_changes = sorted(set(sample))
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)
    return inverted_edf


def make_emb_dist_plot(da_an_dist, da_ad_dist, an_dist_threshold, ax=None):
    # anchor neighbour distance distribution

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    n_peak_widths = 5
    n_bins_till_peak = 5

    xlim_max = an_dist_threshold * n_peak_widths

    kwargs = dict(
        ax=ax,
        range=(0, xlim_max),
        bins=n_bins_till_peak * n_peak_widths,
        histtype="step",
    )
    da_an_dist.plot.hist(label="a-n dist", color="b", **kwargs)
    da_ad_dist.plot.hist(label="a-d dist", color="g", **kwargs)
    ax.legend()

    ax.set_xlim(0, xlim_max)
    ax.set_ylabel("num pairs [1]")

    ax_twin = ax.twinx()
    sns.ecdfplot(da_an_dist, ax=ax_twin, color="b")

    ax_twin.axvline(an_dist_threshold, color="b", linestyle="--")
    sns.ecdfplot(da_ad_dist, ax=ax_twin, color="g")

    return (fig, ax)


def sample_best_triplets(
    x_dim,
    y_dim,
    ds,
    dl=0.1,
    an_dist_max=0.1,
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
    var="emb_pca",
):
    assert "emb" in ds and "an_dist" in ds

    tile_ids_close = ds.an_dist.tile_id.where(ds.an_dist < an_dist_max, drop=True)
    tile_ids_close = tile_ids_close.values.tolist()

    # work out what the dimension of the embedding vector is called
    dim = list((set(ds[var].dims).difference(ds.an_dist.dims)))[0]

    da_x = ds[var].sel({dim: x_dim})
    da_y = ds[var].sel({dim: y_dim})

    # kde = scipy.stats.gaussian_kde([da_x, da_y])
    tile_ids_new = []

    da_x_tile = da_x.sel(tile_id=tile_ids_close)
    da_y_tile = da_y.sel(tile_id=tile_ids_close)

    for x_pca in np.arange(*x_range, dl):
        for y_pca in np.arange(*y_range, dl):
            da_lx = da_x_tile - x_pca
            da_ly = da_y_tile - y_pca

            da_l = np.sqrt(da_lx**2.0 + da_ly**2.0)
            tid = da_l.isel(tile_id=da_l.argmin(dim="tile_id")).tile_id.item()
            if da_l.sel(tile_id=tid) < dl / 2.0:
                tile_ids_new.append(tid)

    tile_ids_new = list(set(tile_ids_new))
    return tile_ids_new


def _get_an_dist(da_embs):
    if "triplet_part" in da_embs.dims:
        da_embs = da_embs.rename(triplet_part="tile_type")

    if "tile_type" not in da_embs.coords:
        raise Exception(
            "To create an isomap embedding plot you need to provide a triplet embeddings. "
            "Expected to find a `tile_type` coordinate, but it it wasn't found"
        )
    da_an_dist = _vector_norm(
        da_embs.sel(tile_type="anchor") - da_embs.sel(tile_type="neighbor"),
        dim="emb_dim",
    )

    return da_an_dist


def _get_anchor_embs_on_manifold(da_embs, method):
    da_embs_anchor = da_embs.sel(tile_type="anchor")

    (
        da_embs_manifold,
        embedding_transform_model,
    ) = embedding_transforms.apply_transform(
        da=da_embs_anchor,
        transform_type=method,
        n_components=2,
        return_model=True,
    )

    return da_embs_manifold, embedding_transform_model


def make_scatter_based_manifold_reference_plot(
    da_embs,
    tile_size=0.02,
    dl=0.1,
    ax=None,
    data_dir="from_embeddings",
    an_dist_ecdf_threshold=0.3,
    method="isomap",
    inset_triplet_distance_distributions=True,
    da_embs_manifold=None,
):
    da_an_dist = _get_an_dist(da_embs=da_embs)
    an_dist_threshold = interp_ecfd(da_an_dist.values)(an_dist_ecdf_threshold)

    if inset_triplet_distance_distributions:
        da_ad_dist = _vector_norm(
            da_embs.sel(tile_type="anchor") - da_embs.sel(tile_type="distant"),
            dim="emb_dim",
        )

    ds = xr.Dataset(
        dict(
            emb=da_embs,
            an_dist=da_an_dist,
        )
    )

    if da_embs_manifold is None:
        da_embs_manifold, embedding_transform_model = _get_anchor_embs_on_manifold(
            da_embs=da_embs, method=method
        )
    else:
        method_precomputed = da_embs_manifold.transform_type
        if method_precomputed != method:
            raise Exception(
                "Method used to produce transformed embeddings in provided file"
                " does not match the method selected for the embedding manifold plot"
                " {method_precomputed} != {method}"
            )
        embedding_transform_model = None

    emb_manifold_var = "emb_anchor_manifold"
    ds[emb_manifold_var] = da_embs_manifold

    tile_ids_sampled = sample_best_triplets(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        x_dim=0,
        y_dim=1,
        ds=ds,
        dl=dl,
        an_dist_max=an_dist_threshold,
        var=emb_manifold_var,
    )

    manifold_dim = f"{method}_dim"
    x = ds[emb_manifold_var].sel({manifold_dim: 0})
    y = ds[emb_manifold_var].sel({manifold_dim: 1})

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 14))
    else:
        fig = ax.figure

    x.attrs["tile_type"] = "ANCHOR"
    y.attrs["tile_type"] = "ANCHOR"
    x.attrs["stage"] = da_embs.stage
    if data_dir == "from_embeddings":
        x.attrs["data_dir"] = da_embs.data_dir
    else:
        x.attrs["data_dir"] = data_dir

    _ = annotated_scatter_plot(
        x=x, y=y, points=tile_ids_sampled, ax=ax, autopos_method=None, size=tile_size
    )

    if inset_triplet_distance_distributions:
        ax_triplet_dists_inset = ax.inset_axes([0.8, 0.8, 0.15, 0.15])
        make_emb_dist_plot(
            da_an_dist=da_an_dist,
            da_ad_dist=da_ad_dist,
            an_dist_threshold=an_dist_threshold,
            ax=ax_triplet_dists_inset,
        )

    return fig, ax, embedding_transform_model


def make_grid_based_manifold_image_slow(
    da_embs_manifold,
    da_an_dist,
    lxy=3.0,
    n_min=16,
    N=16,
    px=32,
    data_dir="from_embeddings",
):
    """
    Create grid-based manifold image. Reference implementation which is quite slow

    lxy: total xy-range
    n_min: minimum number of tiles in bin required to render tile for bin
    N: grid xy-size
    px: number of pixels per image (must be power of 2)
    """
    if data_dir == "from_embeddings":
        data_dir = da_embs_manifold.data_dir

    Nx = Ny = N
    lx = ly = lxy

    dx = lx / Nx
    dy = ly / Ny

    sx = px * Nx
    sy = px * Ny

    img_arr = np.zeros((sx, sy, 4)).astype(np.uint8)
    n_tiles = np.zeros((Nx, Ny))

    da_x = da_embs_manifold.sel(isomap_dim=0)
    da_y = da_embs_manifold.sel(isomap_dim=1)

    xlim = -lx / 2.0, -lx / 2.0 + Nx * dx
    ylim = -ly / 2.0, -ly / 2.0 + Ny * dy

    for i in tqdm(range(Nx), total=Nx):
        for j in range(Ny):
            xmin = -lx / 2.0 + i * dx
            xmax = xmin + dx
            ymin = -ly / 2.0 + j * dy
            ymax = ymin + dy

            mask = (xmin < da_x) * (da_x < xmax) * (ymin < da_y) * (da_y < ymax)
            da_tiles = da_embs_manifold.where(mask, drop=True)
            if da_tiles.count() < n_min:
                continue

            da_an_dist_selected = da_an_dist.sel(tile_id=da_tiles.tile_id)
            da_an_dist_selected = da_an_dist_selected.sortby(da_an_dist_selected)
            da_tile = da_an_dist_selected.isel(tile_id=0)

            try:
                triplet_tile_id = da_tile.triplet_tile_id.item()
            except AttributeError:
                triplet_tile_id = TRIPLET_TILE_IDENTIFIER_FORMAT.format(
                    triplet_id=da_tile.tile_id.item(),
                    tile_type=da_tile.tile_type.item(),
                )
            fp = f"{data_dir}/{da_embs_manifold.stage}/{triplet_tile_id}.png"
            img = Image.open(fp)
            img_arr_raw = np.array(img)
            img_size = img_arr_raw.shape[:2]

            assert img_size[0] == img_size[1]
            step = img_size[0] // px

            img_arr[i * px : (i + 1) * px, j * px : (j + 1) * px] = img_arr_raw[
                ::step, ::step
            ]
            n_tiles[i, j] = da_tiles.count()

    img_manifold = Image.fromarray(np.flipud(np.swapaxes(img_arr, 0, 1)))
    return img_manifold, (xlim, ylim)


def make_grid_based_manifold_image(
    da_embs_manifold,
    da_an_dist,
    lxy=3.0,
    n_min=16,
    N=16,
    px=32,
    pt_c=[0.0, 0.0],
    data_dir="from_embeddings",
):
    """
    Create grid-based manifold image

    lxy: total xy-range
    n_min: minimum number of tiles in bin required to render tile for bin
    N: grid xy-size
    px: number of pixels per image (must be power of 2)
    """
    if data_dir == "from_embeddings":
        data_dir = da_embs_manifold.data_dir

    Nx = Ny = N
    lx = ly = lxy
    dx = lx / Nx
    dy = ly / Ny

    sx = px * Nx
    sy = px * Ny

    # with 4 channels and all zeros the default will be transparent for points
    # in the manifold dimensions without data
    img_arr = np.zeros((sx, sy, 4)).astype(np.uint8)
    n_tiles = np.zeros((Nx, Ny))

    da2 = da_embs_manifold.copy().rename(tile_id="ixy")

    da_x = da2.sel(isomap_dim=0)
    da_y = da2.sel(isomap_dim=1)

    xmin = pt_c[0] - lx / 2.0
    xmax = pt_c[0] + lx / 2.0
    ymin = pt_c[1] - ly / 2.0
    ymax = pt_c[1] + ly / 2.0

    da_ix = ((da_x - xmin) / dx).astype(int)
    da_iy = ((da_y - ymin) / dy).astype(int)

    xlim = xmin, xmax
    ylim = ymin, ymax

    da2.coords["ix"] = da_ix
    da2.coords["iy"] = da_iy
    da2.coords["tile_id"] = da2.ixy
    da2 = da2.set_index(ixy=("ix", "iy"))

    for i in tqdm(range(Nx), total=Nx):
        for j in range(Ny):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    da_tiles = (
                        da2.sel(ixy=(i, j)).swap_dims(ixy="tile_id").drop(["ixy"])
                    )
                    for var in ["ix", "iy"]:
                        if var in da_tiles:
                            da_tiles = da_tiles.drop([var])
            except KeyError:
                continue
            if da_tiles.count() < n_min:
                continue

            da_an_dist_selected = da_an_dist.sel(tile_id=da_tiles.tile_id)
            da_an_dist_selected = da_an_dist_selected.sortby(da_an_dist_selected)

            # load an image, because some of them were spotted to be
            # all-white/all-black (I think because of observation error) we
            # need to check for that so we don't end up with a blank space
            img_arr_raw = None
            nn = 0
            while img_arr_raw is None:
                da_tile = da_an_dist_selected.isel(tile_id=nn)

                try:
                    triplet_tile_id = da_tile.triplet_tile_id.item()
                except AttributeError:
                    triplet_tile_id = TRIPLET_TILE_IDENTIFIER_FORMAT.format(
                        triplet_id=da_tile.tile_id.item(),
                        tile_type=da_tile.tile_type.item(),
                    )
                fp = f"{data_dir}/{da_embs_manifold.stage}/{triplet_tile_id}.png"
                img = Image.open(fp)
                img_arr_raw = np.array(img)

                if len(np.unique(img_arr_raw[:, :, :2])) == 1:
                    if da_tiles.count() < nn:
                        continue
                    img_arr_raw = None
                    nn += 1

            nx_img, ny_img, nc_img = img_arr_raw.shape

            assert nx_img == ny_img
            step = nx_img // px

            sl_x = slice(i * px, (i + 1) * px)
            sl_y = slice(j * px, (j + 1) * px)

            img_arr[sl_x, sl_y, :nc_img] = img_arr_raw[::step, ::step]
            if nc_img == 3:
                # shouldn't be transparent
                img_arr[sl_x, sl_y, 3] = 255

            n_tiles[i, j] = da_tiles.count()

    return Image.fromarray(np.flipud(np.swapaxes(img_arr, 0, 1))), (xlim, ylim)


def make_grid_based_manifold_plot(
    da_embs,
    da_embs_manifold=None,
    ax=None,
    dx=0.05,
    n_min=1,
    px=32,
    pt_c=(0.0, 0.0),
    method="isomap",
    despine_offset=10,
    lxy=None,
    return_image=False,
    **kwargs,
):
    da_an_dist = _get_an_dist(da_embs=da_embs)

    if da_embs_manifold is None:
        da_embs_manifold, embedding_transform_model = _get_anchor_embs_on_manifold(
            da_embs=da_embs, method=method
        )
    else:
        embedding_transform_model = None

    manifold_dim = f"{method}_dim"
    da_x = da_embs_manifold.sel({manifold_dim: 0})
    da_y = da_embs_manifold.sel({manifold_dim: 1})
    if lxy is None:
        d_max = np.max(
            np.abs(
                [
                    pt_c[0] - da_x.min(),
                    da_x.max() - pt_c[0],
                    pt_c[1] - da_y.min(),
                    da_y.max() - pt_c[1],
                ]
            )
        )
        lxy = d_max * 2.0

    # round to resolution
    N = int(lxy / dx)
    lxy = N * dx

    img, (xlim, ylim) = make_grid_based_manifold_image(
        da_embs_manifold=da_embs_manifold,
        da_an_dist=da_an_dist,
        lxy=lxy + 2.0 * dx,  # pad a bit
        n_min=n_min,
        N=N,
        px=px,
        pt_c=pt_c,
        **kwargs,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    lx = ly = lxy
    ax.imshow(img, extent=[*xlim, *ylim])
    ax.set_xlim(pt_c[0] - lx / 2.0, pt_c[0] + lx / 2.0)
    ax.set_ylim(pt_c[1] - ly / 2.0, pt_c[1] + ly / 2.0)
    ax.set_aspect(1)
    if despine_offset:
        sns.despine(offset=despine_offset)
    fig.tight_layout()

    if not return_image:
        return fig, ax, embedding_transform_model
    else:
        return fig, ax, embedding_transform_model, img, (xlim, ylim)


def make_manifold_reference_plot(
    da_embs,
    ax=None,
    method="isomap",
    da_embs_manifold=None,
    plot_type="grid",
    dx=0.05,
    **kwargs,
):
    if plot_type == "scatter":
        return make_scatter_based_manifold_reference_plot(
            ax=ax,
            da_embs=da_embs,
            da_embs_manifold=da_embs_manifold,
            dl=dx,
            method=method,
            **kwargs,
        )
    elif plot_type == "grid":
        return make_grid_based_manifold_plot(
            ax=ax,
            da_embs=da_embs,
            da_embs_manifold=da_embs_manifold,
            dx=dx,
            method=method,
            **kwargs,
        )

    else:
        raise NotImplementedError(plot_type)
