#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import xarray as xr
from sklearn.manifold import Isomap

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

                if kde([x_p, y_p]) < 1.0e-7:
                    continue

                tile_ids_new.append(tid)

    tile_ids_new = list(set(tile_ids_new))
    return tile_ids_new


def make_isomap_reference_plot(da_embs, tile_size=0.02, dl=0.1, ax=None):
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

    model_isomap = Isomap(n_components=2)
    da = ds.emb.sel(tile_type="anchor")
    data = da.values

    ds["emb_isomap"] = xr.DataArray(
        model_isomap.fit_transform(data),
        dims=("tile_id", "isomap_dim"),
        coords=dict(tile_id=da.tile_id),
    )

    tile_ids_sampled = sample_best_triplets(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        x_dim=0,
        y_dim=1,
        ds=ds,
        dl=dl,
        an_dist_max=0.8,
        var="emb_isomap",
    )

    x = ds.emb_isomap.sel(isomap_dim=0)
    y = ds.emb_isomap.sel(isomap_dim=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 14))
    else:
        fig = ax.figure

    x.attrs["tile_type"] = "ANCHOR"
    y.attrs["tile_type"] = "ANCHOR"
    x.attrs["stage"] = da_embs.stage
    x.attrs["data_dir"] = da_embs.data_dir

    _ = annotated_scatter_plot(
        x=x, y=y, points=tile_ids_sampled, ax=ax, autopos_method=None, size=tile_size
    )

    return fig, ax, model_isomap


def plot_embs_on_isomap_manifold(da_embs_triplets, da_embs, dl=0.1, tile_size=0.1):
    if len(da_embs.dims) > 2:
        raise Exception(
            "The embeddings provided should only have a single dimension besides"
            " the embedding dimension (`emb_dim`). Please stack the dimensions"
            f" {da_embs.dims}, e.g. to `(emb_dim, tile_id)`"
        )

    fig, ax, model_isomap = make_isomap_reference_plot(
        da_embs_triplets=da_embs_triplets, dl=dl, tile_size=tile_size
    )

    da_embs_isomap = embedding_transforms.apply_transform(
        da=da_embs,
        transform_type="isomap",
        pretrained_model=model_isomap,
    )
    # ax_overlay = ax.inset_axes([0., 0., 1.0, 1.0], transform=ax.transAxes, zorder=10e9)
    ax_overlay = fig.add_axes(ax.get_position(), zorder=4, transform=ax.transAxes)

    x = da_embs_isomap.sel(isomap_dim=0)
    y = da_embs_isomap.sel(isomap_dim=1)
    ax_overlay.plot(x, y, color="lightgreen", marker=".")
    ax_overlay.plot(x[0], y[0], color="lightgreen", marker="o", markersize=15)

    ax_overlay.patch.set_alpha(0.0)
    ax_overlay.set_xlim(ax.get_xlim())
    ax_overlay.set_ylim(ax.get_ylim())
    ax_overlay.axis("off")

    return fig, ax


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("triplet_embs_path")
    argparser.add_argument("embs_path")
    args = argparser.parse_args()

    filepath_embs = args.embs_path

    da_embs = xr.open_dataarray(filepath_embs)
    da_embs_triplets = xr.open_dataarray(args.triplet_embs_path)

    fig, ax = plot_embs_on_isomap_manifold(
        da_embs_triplets=da_embs_triplets, da_embs=da_embs
    )

    fig.savefig("embs_isomap_overlay.png")
