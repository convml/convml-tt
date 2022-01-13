#!/usr/bin/env python
# coding: utf-8

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import xarray as xr
from sklearn.manifold import Isomap

from ..rectpred import transform as rp_transform
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

    x = ds[var].sel({dim: x_dim})
    y = ds[var].sel({dim: y_dim})

    kde = scipy.stats.gaussian_kde([x, y])
    tile_ids_new = []

    x_ = x.sel(tile_id=tile_ids_close)
    y_ = y.sel(tile_id=tile_ids_close)

    for x_pca in np.arange(*x_range, dl):
        for y_pca in np.arange(*y_range, dl):
            lx = x_ - x_pca
            ly = y_ - y_pca

            l = np.sqrt(lx ** 2.0 + ly ** 2.0)
            tid = l.isel(tile_id=l.argmin(dim="tile_id")).tile_id.item()
            if l.sel(tile_id=tid) < dl:

                x_p, y_p = x_.sel(tile_id=tid), y_.sel(tile_id=tid)

                if kde([x_p, y_p]) < 1.0e-7:
                    continue

                tile_ids_new.append(tid)

    tile_ids_new = list(set(tile_ids_new))
    return tile_ids_new


def _make_isomap_reference_plot(fn_triplet_embeddings):
    da_embs = xr.open_dataarray(fn_triplet_embeddings)

    if "triplet_part" in da_embs.dims:
        da_embs = da_embs.rename(triplet_part="tile_type")

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
        dl=0.1,
        an_dist_max=0.8,
        var="emb_isomap",
    )

    x = ds.emb_isomap.sel(isomap_dim=0)
    y = ds.emb_isomap.sel(isomap_dim=1)

    fig, ax = plt.subplots(figsize=(14, 14))

    x.attrs["tile_type"] = "ANCHOR"
    y.attrs["tile_type"] = "ANCHOR"
    x.attrs["stage"] = "study"
    x.attrs["data_dir"] = da_embs.data_dir

    _ = annotated_scatter_plot(
        x=x, y=y, points=tile_ids_sampled, ax=ax, autopos_method=None, size=0.02
    )

    return fig, ax, model_isomap


def plot_embs_on_isomap_manifold(fn_triplet_embeddings, da_embs):
    fig, ax, model_isomap = _make_isomap_reference_plot(
        fn_triplet_embeddings=fn_triplet_embeddings
    )

    da_embs_isomap = rp_transform.apply_transform(
        da=da_embs * 1.0e0,
        transform_type="isomap",
        pretrained_model=model_isomap,
    )
    # ax_overlay = ax.inset_axes([0., 0., 1.0, 1.0], transform=ax.transAxes, zorder=10e9)
    ax_overlay = fig.add_axes(ax.get_position(), zorder=4, transform=ax.transAxes)

    da_ = da_embs_isomap.isel(x=2, y=2)
    x = da_.sel(isomap_dim=0)
    y = da_.sel(isomap_dim=1)
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

    fig, ax = plot_embs_on_isomap_manifold(
        fn_triplet_embeddings=args.triplet_embs_path, da_embs=da_embs
    )

    fig.savefig("embs_isomap_overlay.png")
