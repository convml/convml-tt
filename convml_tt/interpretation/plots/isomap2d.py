#!/usr/bin/env python
# coding: utf-8

import xarray as xr

from .. import embedding_transforms
from .manifold2d import make_manifold_reference_plot


def make_isomap_reference_plot(
    da_embs,
    tile_size=0.02,
    dl=0.1,
    ax=None,
    data_dir="from_embeddings",
    an_dist_ecdf_threshold=0.3,
    inset_triplet_distance_distributions=True,
):
    """
    Use triplet embeddings in `da_embs` to produce a 2D isomap manifold plot
    picking from the fraction (`an_dist_ecdf_threshold`) of anchor tiles which
    are nearest to the neighbour tile.
    """
    return make_manifold_reference_plot(
        da_embs=da_embs,
        tile_size=tile_size,
        dl=dl,
        ax=ax,
        data_dir=data_dir,
        an_dist_ecdf_threshold=an_dist_ecdf_threshold,
        method="isomap",
        inset_triplet_distance_distributions=inset_triplet_distance_distributions,
    )


def plot_embs_on_isomap_manifold(
    da_embs_triplets,
    da_embs,
    an_dist_ecdf_threshold=0.3,
    tile_size=0.1,
    inset_triplet_distance_distributions=True,
):
    if len(da_embs.dims) > 2:
        raise Exception(
            "The embeddings provided should only have a single dimension besides"
            " the embedding dimension (`emb_dim`). Please stack the dimensions"
            f" {da_embs.dims}, e.g. to `(emb_dim, tile_id)`"
        )

    fig, ax, model_isomap = make_isomap_reference_plot(
        da_embs_triplets=da_embs_triplets,
        an_dist_ecdf_threshold=an_dist_ecdf_threshold,
        tile_size=tile_size,
        method="isomap",
        inset_triplet_distance_distributions=inset_triplet_distance_distributions,
    )

    da_embs_isomap = embedding_transforms.apply_transform(
        da=da_embs,
        transform_type="isomap",
        pretrained_model=model_isomap,
    )
    x = da_embs_isomap.sel(isomap_dim=0)
    y = da_embs_isomap.sel(isomap_dim=1)
    ax.plot(x, y, color="lightgreen", marker=".")
    ax.plot(x[0], y[0], color="lightgreen", marker="o", markersize=15)

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
