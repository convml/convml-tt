"""
Contains `Dataset` definition for loading sets of triplet files for training in
pytorch
"""
import enum
from pathlib import Path

import parse
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch
import numpy as np


class TileType(enum.Enum):
    """Simple enum for mapping into triplet array"""

    ANCHOR = 0
    NEIGHBOR = 1
    DISTANT = 2


class ImageTripletDataset(Dataset):
    """
    image-based dataset for training the triplet-trainer
    """

    TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"
    TRIPLET_META_FILENAME_FORMAT = "{triplet_id:05d}_meta.yaml"

    def __init__(self, data_dir, kind="train", transform=None):
        tile_names = [tile_type.name.lower() for tile_type in TileType]
        # dictionary to hold lists with filepaths for each tile type
        file_paths = {tile_name: [] for tile_name in tile_names}

        ext = self.TILE_FILENAME_FORMAT.split(".")[-1]
        full_path = Path(data_dir) / kind
        for f_path in sorted(full_path.glob(f"*.{ext}"), key=lambda p: p.name):
            file_info = parse.parse(self.TILE_FILENAME_FORMAT, f_path.name)
            tile_name = file_info["tile_type"]

            if tile_name in tile_names:
                file_paths[tile_name].append(f_path)

        n_tiles = {name: len(files) for (name, files) in file_paths.items()}
        if not len(set(n_tiles.values())) == 1:
            raise Exception(
                f"A different number of tiles of each type were found ({n_tiles})"
            )
        self.num_items = list(n_tiles.values())[0]

        if set(n_tiles.values()) == 0:
            raise Exception(f"No {kind} data was found")

        self.file_paths = file_paths
        self.transform = transform

    def _read_image(self, index, tile_type):
        single_image_path = self.file_paths[tile_type][index]
        im_as_im = Image.open(single_image_path)
        return im_as_im

    def __getitem__(self, index):
        item_contents = [
            self._read_image(index, tile_type.name.lower())
            for tile_type in TileType
        ]
        if self.transform:
            item_contents = [self.transform(v) for v in item_contents]
        return item_contents

    def __len__(self):
        return self.num_items
