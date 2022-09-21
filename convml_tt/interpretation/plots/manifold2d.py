#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.distributions.empirical_distribution as edf
import xarray as xr
from scipy.interpolate import interp1d

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


def make_manifold_reference_plot(
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

    manifold_dim = f"{method}_dim"
    emb_manifold_var = "emb_anchor_manifold"

    if da_embs_manifold is None:
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
    else:
        method_precomputed = da_embs_manifold.transform_type
        if method_precomputed != method:
            raise Exception(
                "Method used to produce transformed embeddings in provided file"
                " does not match the method selected for the embedding manifold plot"
                " {method_precomputed} != {method}"
            )
        embedding_transform_model = None

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
