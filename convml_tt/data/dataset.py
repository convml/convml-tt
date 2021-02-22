"""
Contains `Dataset` definition for loading sets of triplet files for training in
pytorch
"""
import enum
from pathlib import Path

import parse
from torch.utils.data.dataset import Dataset
from PIL import Image


class TileType(enum.Enum):
    """Simple enum for mapping into triplet array"""

    ANCHOR = 0
    NEIGHBOR = 1
    DISTANT = 2


class _ImageDatasetBase(Dataset):

    TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"

    def _find_files(self):
        tile_names = [tile_type.name.lower() for tile_type in TileType]
        # dictionary to hold lists with filepaths for each tile type
        file_paths = {tile_name: [] for tile_name in tile_names}

        ext = self.TILE_FILENAME_FORMAT.split(".")[-1]
        for f_path in sorted(self.full_path.glob(f"*.{ext}"), key=lambda p: p.name):
            file_info = parse.parse(self.TILE_FILENAME_FORMAT, f_path.name)
            tile_name = file_info["tile_type"]

            if tile_name in tile_names:
                file_paths[tile_name].append(f_path)
        return file_paths

    def __init__(self, data_dir, stage="train", transform=None):
        self.transform = transform
        self.num_items = -1
        self.full_path = Path(data_dir) / stage

    def _read_image(self, single_image_path):
        im_as_im = Image.open(single_image_path)
        return im_as_im

    def __len__(self):
        return self.num_items


class ImageTripletDataset(_ImageDatasetBase):
    """
    image-based dataset for training the triplet-trainer
    """
    TRIPLET_META_FILENAME_FORMAT = "{triplet_id:05d}_meta.yaml"

    def __init__(self, data_dir, stage="train", transform=None):
        super().__init__(data_dir=data_dir, stage=stage, transform=transform)

        self.file_paths = self._find_files()
        n_tiles = {name: len(files) for (name, files) in self.file_paths.items()}
        if not len(set(n_tiles.values())) == 1:
            raise Exception(
                f"A different number of tiles of each type were found ({n_tiles})"
            )
        self.num_items = list(n_tiles.values())[0]

        if set(n_tiles.values()) == 0:
            raise Exception(f"No {stage} data was found")

    def __getitem__(self, index):
        item_contents = [
            self._read_image(self.file_paths[tile_type.name.lower()][index])
            for tile_type in TileType
        ]
        if self.transform:
            item_contents = [self.transform(v) for v in item_contents]
        return item_contents


class ImageSingletDataset(_ImageDatasetBase):
    def __init__(self, data_dir, tile_type: TileType, stage="train", transform=None):
        super().__init__(data_dir=data_dir, stage=stage, transform=transform)

        self.file_paths = self._find_files()[tile_type.name.lower()]
        self.num_items = len(self.file_paths)

        if self.num_items == 0:
            raise Exception(f"No {stage} data was found")

    def __getitem__(self, index):
        image_file_path = self.file_paths[index]
        item = self._read_image(image_file_path)

        if self.transform:
            item = self.transform(item)
        return item
