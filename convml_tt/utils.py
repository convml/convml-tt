import xarray as xr
import numpy as np
import torch
from tqdm import tqdm

from pathlib import Path

from PIL import Image

from .architectures.triplet_trainer import TileType

def get_encodings(triplets, model, tile_type=TileType.ANCHOR):
    def _get_encoding(image_set, *args, **kwargs):
        try:
            v = model.predict(image_set[tile_type])[1]
        except RuntimeError:
            # "... eturns a tuple of three things: the object predicted (with
            # the class in this instance), the underlying data (here the
            # corresponding index) and the raw probabilities"
            # https://docs.fast.ai/tutorial.inference.html#Vision
            # we only want the underlying data (the three embeddings for the
            # triplet provided)
            v = model.predict(image_set)[1][tile_type]
        return image_set.id, v

    triplet_ids_and_encodings = [
        _get_encoding(image_set) for image_set in tqdm(triplets)
    ]

    triplet_ids, encodings = zip(*triplet_ids_and_encodings)

    encodings = np.asarray(torch.stack(encodings).squeeze())
    # we're picking out the anchor tile above
    tile_id = np.asarray(triplet_ids).astype(int)

    return xr.DataArray(
        encodings, dims=('tile_id', 'enc_dim'),
        coords=dict(tile_id=tile_id, enc_dim=np.arange(encodings.shape[1])),
        attrs=dict(tile_used=TileType.NAMES[tile_type],
        source_path=str(triplets.path.absolute())
        )
    )

class ImageLoader:
    def __init__(self, path):
        self.path = Path(path)

    def __getitem__(self, n):
        img_path = self.path/"{:05d}_anchor.png".format(n)
        return Image.open(img_path)

def get_triplets_from_encodings(encodings):
    return ImageLoader(encodings.source_path)
