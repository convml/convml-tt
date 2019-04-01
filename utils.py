import xarray as xr
import numpy as np
import torch
from tqdm import tqdm

from pathlib import Path

from PIL import Image

def get_encodings(triplets, model):
    tile_used = (0, 'anchor')
    def _get_encoding(image_set, *args, **kwargs):
        v = model.predict(image_set[tile_used[0]])[1]
        return image_set.id, v

    triplet_ids_and_encodings = [
        _get_encoding(image_set) for image_set in tqdm(triplets)
    ]

    triplet_ids, encodings = zip(*triplet_ids_and_encodings)

    encodings = np.asarray(torch.stack(encodings))
    # we're picking out the anchor tile above
    tile_id = np.asarray(triplet_ids)

    return xr.DataArray(
        encodings, dims=('tile_id', 'enc_dim'),
        coords=dict(tile_id=tile_id),
        attrs=dict(tile_used=tile_used[1],
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
