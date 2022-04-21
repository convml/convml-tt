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
    anchor_neighbor_max_dist=0.1,
    min_point_density=1.0e-3,
):
    make_manifold_reference_plot(
        da_embs=da_embs,
        tile_size=tile_size,
        dl=dl,
        ax=ax,
        data_dir=data_dir,
        anchor_neighbor_max_dist=anchor_neighbor_max_dist,
        min_point_density=min_point_density,
        method="isomap",
    )


def plot_embs_on_isomap_manifold(da_embs_triplets, da_embs, dl=0.1, tile_size=0.1):
    if len(da_embs.dims) > 2:
        raise Exception(
            "The embeddings provided should only have a single dimension besides"
            " the embedding dimension (`emb_dim`). Please stack the dimensions"
            f" {da_embs.dims}, e.g. to `(emb_dim, tile_id)`"
        )

    fig, ax, model_isomap = make_isomap_reference_plot(
        da_embs_triplets=da_embs_triplets,
        dl=dl,
        tile_size=tile_size,
        method="isomap",
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
