"""
Contains `Dataset` definition for loading sets of triplet files for training in
pytorch
"""
import enum
from pathlib import Path

import parse
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class TileType(enum.Enum):
    """Simple enum for mapping into triplet array"""

    ANCHOR = 0
    NEIGHBOR = 1
    DISTANT = 2


class _ImageDatasetBase(Dataset):
    TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.{ext}"

    def _find_files(self):
        # dictionary to hold lists with filepaths for each tile type
        file_paths = {tile_type: [] for tile_type in TileType}

        ext = self.TILE_FILENAME_FORMAT.split(".")[-1]
        full_path = Path(self.data_dir) / self.stage
        for f_path in sorted(full_path.glob(f"*.{ext}"), key=lambda p: p.name):
            file_info = parse.parse(self.TILE_FILENAME_FORMAT, f_path.name)
            tile_name = file_info["tile_type"]
            try:
                tile_type = TileType[tile_name.upper()]
                file_paths[tile_type].append(f_path)
            except KeyError:
                pass
        return file_paths

    def __init__(self, data_dir, stage="train", transform=None):
        self.transform = transform
        self.num_items = -1
        self.data_dir = data_dir
        self.stage = stage

    def _read_image(self, single_image_path):
        im_as_im = Image.open(single_image_path)
        return im_as_im

    def __len__(self):
        return self.num_items


class ImageTripletDataset(_ImageDatasetBase):
    """
    image-based dataset for training the triplet-trainer
    """


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

    def get_image(self, index, tile_type):
        image_file_path = self.file_paths[tile_type][index]
        return self._read_image(image_file_path)

    def __getitem__(self, index):
        item_contents = [
            self._read_image(self.file_paths[tile_type][index])
            for tile_type in TileType
        ]
        if self.transform:
            item_contents = [self.transform(v) for v in item_contents]

        return item_contents


class ImageSingletDataset(_ImageDatasetBase):
    def __init__(
        self,
        data_dir,
        tile_type: TileType,
        stage="train",
        transform=None,
    ):
        super().__init__(data_dir=data_dir, stage=stage, transform=transform)

        if type(tile_type) == str:
            tile_type = TileType[tile_type]

        self.file_paths = self._find_files()[tile_type]
        self.num_items = len(self.file_paths)
        self.tile_type = tile_type

        if self.num_items == 0:
            raise Exception(f"No {stage} data was found")

    def get_image(self, index):
        image_file_path = self.file_paths[index]
        return self._read_image(image_file_path)

    def __getitem__(self, index):
        image_file_path = self.file_paths[index]
        item = self._read_image(image_file_path)

        if self.transform:
            item = self.transform(item)
        return item
