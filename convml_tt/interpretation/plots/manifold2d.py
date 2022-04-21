#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import xarray as xr

from .. import embedding_transforms
from . import annotated_scatter_plot


def _vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(
        np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
    )


def sample_best_triplets(
    x_dim,
    y_dim,
    ds,
    dl=0.1,
    an_dist_max=0.1,
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
    var="emb_pca",
    min_point_density=1.0e-3,
):
    assert "emb" in ds and "an_dist" in ds

    tile_ids_close = ds.an_dist.tile_id.where(ds.an_dist < an_dist_max, drop=True)
    tile_ids_close = tile_ids_close.values.tolist()

    bad_tiles = [8801, 4779, 8498]
    for tile_id_bad in bad_tiles:
        if tile_id_bad in tile_ids_close:
            tile_ids_close.remove(tile_id_bad)

    # work out what the dimension of the embedding vector is called
    dim = list((set(ds[var].dims).difference(ds.an_dist.dims)))[0]

    da_x = ds[var].sel({dim: x_dim})
    da_y = ds[var].sel({dim: y_dim})

    kde = scipy.stats.gaussian_kde([da_x, da_y])
    tile_ids_new = []

    da_x_tile = da_x.sel(tile_id=tile_ids_close)
    da_y_tile = da_y.sel(tile_id=tile_ids_close)

    for x_pca in np.arange(*x_range, dl):
        for y_pca in np.arange(*y_range, dl):
            da_lx = da_x_tile - x_pca
            da_ly = da_y_tile - y_pca

            da_l = np.sqrt(da_lx ** 2.0 + da_ly ** 2.0)
            tid = da_l.isel(tile_id=da_l.argmin(dim="tile_id")).tile_id.item()
            if da_l.sel(tile_id=tid) < dl:

                x_p, y_p = da_x_tile.sel(tile_id=tid), da_y_tile.sel(tile_id=tid)

                if kde([x_p, y_p]) < min_point_density:
                    continue

                tile_ids_new.append(tid)

    tile_ids_new = list(set(tile_ids_new))
    return tile_ids_new


def make_manifold_reference_plot(
    da_embs,
    tile_size=0.02,
    dl=0.1,
    ax=None,
    data_dir="from_embeddings",
    anchor_neighbor_max_dist=0.1,
    min_point_density=1.0e-3,
    method="isomap",
):
    if "triplet_part" in da_embs.dims:
        da_embs = da_embs.rename(triplet_part="tile_type")

    if "tile_type" not in da_embs.coords:
        raise Exception(
            "To create an isomap embedding plot you need to provide a triplet embeddings. "
            "Expected to find a `tile_type` coordinate, but it it wasn't found"
        )

    da_embs_neardiff = da_embs.sel(tile_type="anchor") - da_embs.sel(
        tile_type="neighbor"
    )
    da_embs_neardiff_mag = _vector_norm(da_embs_neardiff, dim="emb_dim")

    ds = xr.Dataset(
        dict(
            emb=da_embs,
            an_dist=da_embs_neardiff_mag,
        )
    )

    da_embs_anchor = da_embs.sel(tile_type="anchor")

    # da_embs_anchor = embedding_transforms.apply_transform(
    # da_embs_anchor, transform_type="standard_scaler"
    # ).rename(dict(standard_scaler_dim="emb_dim"))

    emb_manifold_var = "emb_anchor_manifold"
    manifold_dim = f"{method}_dim"

    (
        ds[emb_manifold_var],
        embedding_transform_model,
    ) = embedding_transforms.apply_transform(
        da=da_embs_anchor,
        transform_type=method,
        n_components=2,
        return_model=True,
    )

    tile_ids_sampled = sample_best_triplets(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        x_dim=0,
        y_dim=1,
        ds=ds,
        dl=dl,
        an_dist_max=anchor_neighbor_max_dist,
        var=emb_manifold_var,
        min_point_density=min_point_density,
    )

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

    return fig, ax, embedding_transform_model
