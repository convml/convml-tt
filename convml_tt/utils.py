"""
Utility functions for the triplet-trainer
"""
from pathlib import Path

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import ImageSingletDataset, ImageTripletDataset


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
