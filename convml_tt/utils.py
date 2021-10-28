"""
Utility functions for the triplet-trainer
"""
from pathlib import Path

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import (
    ImageSingletDataset,
    ImageTripletDataset,
    MovingWindowImageTilingDataset,
)


def get_embeddings(
    tile_dataset: [ImageSingletDataset, ImageTripletDataset],
    model,
    prediction_batch_size=32,
):
    """
    Use the provided model to calculate the embeddings for all tiles of a
    specific `tile_type` in the given `data_dir`. If you run out of memory
    reduce the `prediction_batch_size` (you may also increase it to generate
    predictions faster while using more RAM).
    """
    tile_dataloader = DataLoader(dataset=tile_dataset, batch_size=prediction_batch_size)
    batched_results = []

    def apply_model(x):
        return model.forward(x).cpu().detach().numpy()

    tile_ids = np.arange(len(tile_dataloader.dataset))
    coords = dict(tile_id=tile_ids)

    if isinstance(tile_dataset, ImageSingletDataset):
        for x_batch in tqdm(tile_dataloader):
            y_batch = apply_model(x_batch)
            batched_results.append(y_batch)
        dims = ("tile_id", "emb_dim")
    elif isinstance(tile_dataset, ImageTripletDataset):
        triplet_batch = []
        for xs_batch in tqdm(tile_dataloader):
            ys_batch = [apply_model(x_batch) for x_batch in xs_batch]
            triplet_batch.append(ys_batch)
        batched_results.append(np.hstack(triplet_batch))
        dims = ("tile_type", "tile_id", "emb_dim")
        coords["tile_type"] = ["anchor", "neighbor", "distant"]

    embeddings = np.vstack(batched_results)

    attrs = {}
    if hasattr(tile_dataset, "tile_type"):
        attrs["tile_type"] = tile_dataset.tile_type.name
    if hasattr(tile_dataset, "stage"):
        attrs["stage"] = tile_dataset.stage
    if hasattr(tile_dataset, "data_dir"):
        attrs["data_dir"] = str(Path(tile_dataset.data_dir).absolute())

    return xr.DataArray(embeddings, dims=dims, coords=coords, attrs=attrs)


def make_sliding_tile_model_predictions(
    model,
    tile_dataset: MovingWindowImageTilingDataset,
    prediction_batch_size=32,
):
    """
    Produce moving-window prediction array from at moving-window tiling dataset
    with `model`.

    NB: j-indexing is from "top_left" i.e. is likely in the opposite order to
    what would be expected for y-axis of original image (positive being up)

    i0 and j0 coordinates denote center of each prediction tile
    """
    da_emb = get_embeddings(
        tile_dataset=tile_dataset,
        model=model,
        prediction_batch_size=prediction_batch_size,
    )

    # "unstack" the 2D array with coords (tile_id, emb_dim) to have coords (i0,
    # j0, emb_dim), where `i0` and `j0` represent the index of the pixel in the
    # original image at the center of each tile
    i_img_tile, j_img_tile = tile_dataset.index_to_img_ij(da_emb.tile_id.values)
    i_img_tile_center = (i_img_tile + 0.5 * tile_dataset.nxt).astype(int)
    j_img_tile_center = (j_img_tile + 0.5 * tile_dataset.nyt).astype(int)
    da_emb["i0"] = ("tile_id"), i_img_tile_center
    da_emb["j0"] = ("tile_id"), j_img_tile_center
    da_emb = da_emb.set_index(tile_id=("i0", "j0")).unstack("tile_id")

    da_emb.attrs["tile_nx"] = tile_dataset.nxt
    da_emb.attrs["tile_ny"] = tile_dataset.nyt

    return da_emb
