"""
Utility functions for the triplet-trainer
"""
from pathlib import Path

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import ImageSingletDataset


def get_embeddings(tile_dataset: ImageSingletDataset, model, prediction_batch_size=32):
    """
    Use the provided model to calculate the embeddings for all tiles of a
    specific `tile_type` in the given `data_dir`. If you run out of memory
    reduce the `prediction_batch_size` (you may also increase it to generate
    predictions faster while using more RAM).
    """
    tile_dataloader = DataLoader(dataset=tile_dataset, batch_size=prediction_batch_size)
    batched_results = []
    for x_batch in tqdm(tile_dataloader):
        y_batch = model.forward(x_batch)
        batched_results.append(y_batch.cpu().detach().numpy())

    embeddings = np.vstack(batched_results)
    tile_ids = np.arange(len(tile_dataloader.dataset))

    dims = ("tile_id", "emb_dim")
    coords = dict(tile_id=tile_ids)

    attrs = {}
    if hasattr(tile_dataset, "tile_type"):
        attrs["tile_type"] = tile_dataset.tile_type.name
    if hasattr(tile_dataset, "stage"):
        attrs["stage"] = tile_dataset.stage
    if hasattr(tile_dataset, "data_dir"):
        attrs["data_dir"] = str(Path(tile_dataset.data_dir).absolute())

    return xr.DataArray(embeddings, dims=dims, coords=coords, attrs=attrs)
