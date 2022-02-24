"""
Contains `Dataset` definition for loading sets of triplet files for training in
pytorch
"""
import enum
from pathlib import Path

import numpy as np
import parse
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms as tv_transforms
from tqdm import tqdm

from .common import TILE_IDENTIFIER_FORMAT


def get_load_transforms():
    return tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            RemoveImageAlphaTransform(),
        ]
    )


class RemoveImageAlphaTransform:
    def __call__(self, x):
        return x[:3, :, :]


class TileType(enum.Enum):
    """Simple enum for mapping into triplet array"""

    ANCHOR = 0
    NEIGHBOR = 1
    DISTANT = 2


def _find_tile_files(data_dir, stage, ext="png"):
    # dictionary to hold lists with filepaths for each tile type
    file_paths = {tile_type: [] for tile_type in TileType}

    full_path = Path(data_dir) / stage
    for f_path in sorted(full_path.glob(f"*.{ext}"), key=lambda p: p.name):
        file_info = parse.parse(TILE_IDENTIFIER_FORMAT + f".{ext}", f_path.name)
        tile_name = file_info["tile_type"]
        try:
            tile_type = TileType[tile_name.upper()]
            file_paths[tile_type].append(f_path)
        except KeyError:
            pass
    return file_paths


class _ImageDatasetBase(Dataset):
    def __init__(self, data_dir, stage="train", transform=None):
        self.transform = transform
        self.num_items = -1
        self.data_dir = data_dir
        self.stage = stage

        # prepare the transforms that ensure we get from an image to a torch
        # tensor
        self._image_load_transforms = get_load_transforms()

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

        self.file_paths = _find_tile_files(data_dir=data_dir, stage=stage)
        n_tiles = {name: len(files) for (name, files) in self.file_paths.items()}
        if not len(set(n_tiles.values())) == 1:
            raise Exception(
                f"A different number of tiles of each type were found ({n_tiles})"
            )
        self.num_items = list(n_tiles.values())[0]

        if set(n_tiles.values()) == {0}:
            raise FileNotFoundError(f"No {stage} data was found in `{data_dir}`")

    def make_singlet_dataset(self, tile_type):
        """Produce an ImageSingletDataset for a particular tile type"""
        return ImageSingletDataset(
            data_dir=self.data_dir,
            tile_type=tile_type,
            stage=self.stage,
            transform=self.transform,
        )

    def get_image(self, index, tile_type):
        image_file_path = self.file_paths[tile_type][index]
        return self._read_image(image_file_path)

    def _get_image_tensor(self, index, tile_type):
        return self._image_load_transforms(
            self.get_image(index=index, tile_type=tile_type)
        )

    def __getitem__(self, index):
        item_contents = [
            self._get_image_tensor(index=index, tile_type=tile_type)
            for tile_type in TileType
        ]
        if self.transform:
            item_contents = [self.transform(v) for v in item_contents]

        return item_contents


class MemoryMappedImageTripletDataset(ImageTripletDataset):
    """
    Uses a memory-mapped file to store image data as an intermediate numpy
    array, which should speed loading up as the images will only be loaded once
    from disc
    """

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir, *args, **kwargs)

        stage = kwargs["stage"]
        self.mm_filepath = Path(data_dir) / f"{stage}_cache.dat"
        self._preload_all_tile_image_data()

    def _preload_all_tile_image_data(self):
        # load the first image to get the shape
        img0 = self.get_image(index=0, tile_type=TileType.ANCHOR)
        img0_arr = np.array(img0)
        (nx, ny, _) = img0_arr.shape
        nc = 3
        n_samples = len(self)
        n_tiles_per_sample = len(TileType)

        mm_shape = (n_samples, n_tiles_per_sample, nx, ny, nc)
        mm_dtype = "float32"

        if not self.mm_filepath.exists():

            self._data = np.memmap(
                str(self.mm_filepath),
                dtype=mm_dtype,
                mode="w+",
                shape=mm_shape,
            )

            for i in tqdm(range(n_samples), desc="creating memory-mapped file"):
                for t_i, tile_type in enumerate(TileType):
                    img_data = self.get_image(index=i, tile_type=tile_type)
                    # make sure we strip of the alpha channel if it is there
                    self._data[i, t_i] = np.array(img_data)[:, :, :nc]

            # Flush memory changes to disk in order to read them back
            self._data.flush()
        else:
            self._data = np.array(
                np.memmap(
                    str(self.mm_filepath), dtype=mm_dtype, mode="r", shape=mm_shape
                )
            )

    def __getitem__(self, index):
        item_contents = [
            self._image_load_transforms(self._data[index, t_i, :, :, :])
            for t_i, tile_type in enumerate(TileType)
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

        self.file_paths = _find_tile_files(data_dir=data_dir, stage=stage)[tile_type]
        self.num_items = len(self.file_paths)
        self.tile_type = tile_type

        if self.num_items == 0:
            raise Exception(f"No {stage} data was found")

    def get_image(self, index):
        image_file_path = self.file_paths[index]
        return self._read_image(image_file_path)

    def __getitem__(self, index):
        item = self._image_load_transforms(self.get_image(index=index))

        if self.transform:
            item = self.transform(item)
        return item


class MovingWindowImageTilingDataset(ImageSingletDataset):
    def __init__(
        self,
        img,
        transform=None,
        step=(50, 50),
        N_tile=(256, 256),
    ):
        """
        Produce moving-window iling dataset with with step-size defined by
        `step` and tile-size `N_tile`.
        """

        super(ImageSingletDataset).__init__()
        self.nx, self.ny = img.size
        self.nxt, self.nyt = N_tile
        self.x_step, self.y_step = step
        self.img = img

        # the starting image x-index for tile with tile x-index `i`
        self.img_idx_tile_i = np.arange(0, self.nx - self.nxt + 1, self.x_step)
        # the starting image y-index for tile with tile y-index `j`
        self.img_idx_tile_j = np.arange(0, self.ny - self.nyt + 1, self.y_step)

        # number of tiles in x- and y-direction
        self.nt_x = len(self.img_idx_tile_i)
        self.nt_y = len(self.img_idx_tile_j)

        self.num_items = self.nt_x * self.nt_y

        image_load_transforms = get_load_transforms()
        self.img_data_normed = image_load_transforms(self.img)
        self.transform = transform

    def index_to_img_ij(self, index):
        """
        Turn the tile_id (running from 0 to the number of tiles) into the
        (i,j)-index for the tile
        """
        # tile-index in (i,j) running for the number of tiles in each direction
        j = index // self.nt_x
        i = index - self.nt_x * j

        # indecies into tile shape
        i_img = self.img_idx_tile_i[i]
        j_img = self.img_idx_tile_j[j]
        return i_img, j_img

    def get_image(self, index):
        x_slice, y_slice = self._get_image_tiling_slices(index=index)
        # for a PIL image we need the unnormalized image data
        img_data = np.array(self.img)
        # PIL images have shape [height, width, #channels]
        img_data_tile = img_data[y_slice, x_slice, :]
        return Image.fromarray(img_data_tile)

    def _get_image_tiling_slices(self, index):
        i_img, j_img = self.index_to_img_ij(index=index)

        x_slice = slice(i_img, i_img + self.nxt)
        y_slice = slice(j_img, j_img + self.nyt)
        return (x_slice, y_slice)

    def __getitem__(self, index):
        x_slice, y_slice = self._get_image_tiling_slices(index=index)
        img_data_tile = self.img_data_normed[:, y_slice, x_slice]

        return self.transform(img_data_tile)
