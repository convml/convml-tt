"""
Contains `Dataset` definition for loading sets of triplet files for training in
pytorch
"""
import enum
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import parse
import xarray as xr
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torchvision import transforms as tv_transforms
from tqdm import tqdm

from .common import TRIPLET_TILE_IDENTIFIER_FORMAT

# try and handle `OSError: image file is truncated`
# https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


TRIPLET_TILE_FILENAME_FORMAT = f"{TRIPLET_TILE_IDENTIFIER_FORMAT}.png"


def _find_tile_files(
    data_path,
    tile_identifier_format=TRIPLET_TILE_IDENTIFIER_FORMAT,
    ext="png",
    read_meta_from_netcdf_source="if_available",
    progress=False,
    cache_to_csv=True,
    aux_accessor=None,
):
    """
    Find all tile files in `data_path` which match the filename
    format `{tile_identifier_format}.{ext}` and return a `pandas.DataFrame`
    with the filenames and the meta information
    parsed from the filename(s). If `read_meta_from_netcdf_source` is `True`
    then the meta information from the source netCDF files will also
    be included. Unless `cache_to_csv` is `False` the result will be stored
    in a CSV file called `meta.csv` in `data_path`.
    """
    data_path = Path(data_path)
    fpaths_ext = [
        fp.relative_to(data_path)
        for fp in sorted(data_path.glob(f"*.{ext}"), key=lambda p: p.name)
    ]

    df_tiles = None
    fpath_csv = data_path / "meta.csv"
    if cache_to_csv and fpath_csv.exists():
        if cache_to_csv == "overwrite":
            fpath_csv.unlink()
        else:
            df_tiles = pd.read_csv(fpath_csv)

    if progress:
        progress_fn = tqdm
    else:
        progress_fn = lambda x: x

    # attempt to parse meta information from filenames of all files
    # for those where the filename doesn't match the correct format
    # the result of `parse.parse` will be None
    fpaths_meta = {}
    for fpath in progress_fn(fpaths_ext):
        filename_meta = parse.parse(tile_identifier_format + f".{ext}", fpath.name)
        if filename_meta is not None:
            fpaths_meta[fpath] = filename_meta.named

    # if we're loading from a CSV file check that it mentions the
    # same files
    if df_tiles is not None:
        expected_fpaths = set([str(fp) for fp in fpaths_meta.keys()])
        actual_fpaths = set(df_tiles.filepath)
        difference_fpaths = expected_fpaths.difference(actual_fpaths)
        if len(difference_fpaths) == 0:
            return df_tiles
        else:
            warnings.warn(
                "Stored CSV of meta information doesn't match the files in "
                "the provided `data_path. Recreating meta info CSV file"
            )
            df_tiles = None

    # parse meta info from the filenames (and optionally the source
    # netcdf files)
    tiles_meta = {"filepath": []}
    for fpath, filename_meta in progress_fn(fpaths_meta.items()):
        if filename_meta is not None:
            tiles_meta["filepath"].append(fpath)
            tiles_meta
            for meta_field, meta_value in filename_meta.items():
                tiles_meta.setdefault(meta_field, []).append(meta_value)

            if read_meta_from_netcdf_source:
                fpath_netcdf = fpath.parent / fpath.name.replace(f".{ext}", ".nc")

                ds_src = None
                try:
                    ds_src = xr.open_dataset(fpath_netcdf)
                except Exception:
                    if read_meta_from_netcdf_source == "if_available":
                        pass
                    else:
                        raise

                if ds_src is not None:
                    if len(ds_src.data_vars) == 1:
                        src_attrs = ds_src[list(ds_src.data_vars)[0]].attrs
                    else:
                        src_attrs = ds_src.attrs

                    for meta_field, meta_value in src_attrs.items():
                        tiles_meta.setdefault(meta_field, []).append(meta_value)

            if aux_accessor is not None:
                for k, v in aux_accessor(fpath).items():
                    tiles_meta.setdefault(k, []).append(v)

    df_tiles = pd.DataFrame(tiles_meta)
    if cache_to_csv:
        df_tiles.to_csv(fpath_csv, index=False)
    return df_tiles


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
        return len(self.df_tiles.index)


class ImageTripletDataset(_ImageDatasetBase):
    """
    image-based dataset for training the triplet-trainer
    """

    TRIPLET_META_FILENAME_FORMAT = "{triplet_id:05d}_meta.yaml"

    def __init__(
        self, data_dir, stage="train", transform=None, skip_incomplete_triplets=False
    ):
        super().__init__(data_dir=data_dir, stage=stage, transform=transform)

        if stage is not None:
            self.tiles_data_path = Path(data_dir) / stage
        else:
            self.tiles_data_path = Path(data_dir)

        df_tiles_singles = _find_tile_files(
            self.tiles_data_path, tile_identifier_format=TRIPLET_TILE_IDENTIFIER_FORMAT
        )
        # pivot so that we group the tiles by triplet
        self.df_tiles = df_tiles_singles.pivot(index="triplet_id", columns="tile_type")

        # make the index be called "tile_id" rather than "triplet_id"
        self.df_tiles = self.df_tiles.rename(columns=dict(triplet_id="tile_id"))

        df_missing_triplet_parts = self.df_tiles[self.df_tiles.isnull().any(axis=1)]
        if df_missing_triplet_parts.count().max() > 0:
            if not skip_incomplete_triplets:
                raise Exception(
                    f"Some triplets don't have all three tiles: {df_missing_triplet_parts}"
                )
            else:
                raise NotImplementedError()

        if len(self) == 0:
            stage_s = stage is not None and f" {stage}" or ""
            raise FileNotFoundError(f"No {stage_s} data was found in `{data_dir}`")

    def make_singlet_dataset(self, tile_type):
        """Produce an ImageSingletDataset for a particular tile type"""
        return ImageSingletDataset(
            data_dir=self.data_dir,
            tile_type=tile_type,
            stage=self.stage,
            transform=self.transform,
        )

    def get_image(self, tile_id, tile_type):
        tile_type_s = tile_type.name.lower()
        image_file_path = (
            self.tiles_data_path / self.df_tiles["filepath"][tile_type_s].loc[tile_id]
        )
        return self._read_image(image_file_path)

    def _get_image_tensor(self, index, tile_type):
        tile_id = self.df_tiles.index[index]
        return self._image_load_transforms(
            self.get_image(tile_id=tile_id, tile_type=tile_type)
        )

    def __getitem__(self, index):
        item_contents = [
            self._get_image_tensor(index=index, tile_type=tile_type)
            for tile_type in TileType
        ]
        if self.transform:
            item_contents = [self.transform(v) for v in item_contents]

        return item_contents

    @property
    def index(self):
        return self.df_tiles.index


class ImageSingletDataset(_ImageDatasetBase):
    def __init__(
        self,
        data_dir,
        tile_type: TileType,
        stage="train",
        transform=None,
        tile_identifier_format=TRIPLET_TILE_IDENTIFIER_FORMAT,
        filter_func=None,
    ):
        super().__init__(data_dir=data_dir, stage=stage, transform=transform)

        if "triplet_id" in tile_identifier_format and tile_type is None:
            raise Exception(
                "You must select a tile-type when creating a singlet dataset from"
                " a dataset made of triplets"
            )

        if tile_type is not None and type(tile_type) == str:
            tile_type = TileType[tile_type]

        if stage is not None:
            self.tiles_data_path = Path(data_dir) / stage
        else:
            self.tiles_data_path = Path(data_dir)

        self.df_tiles = _find_tile_files(
            self.tiles_data_path, tile_identifier_format=tile_identifier_format
        )

        if tile_type is not None:
            self.df_tiles = (
                self.df_tiles[self.df_tiles.tile_type == str(tile_type.name.lower())]
                .rename(columns=dict(triplet_id="tile_id"))
                .set_index("tile_id")
            )
        else:
            self.df_tiles = self.df_tiles.set_index("tile_id")
        self.tile_type = tile_type

        if filter_func is not None:
            self.df_tiles = filter_func(df_tiles=self.df_tiles)

        if len(self) == 0:
            stage_s = stage is not None and f" {stage}" or ""
            raise FileNotFoundError(f"No {stage_s} data was found in `{data_dir}`")

    def get_image(self, tile_id):
        image_file_path = self.tiles_data_path / self.df_tiles.filepath.loc[tile_id]
        return self._read_image(image_file_path)

    def __getitem__(self, index):
        tile_id = self.df_tiles.index[index]
        item = self._image_load_transforms(self.get_image(tile_id=tile_id))

        if self.transform:
            item = self.transform(item)
        return item

    @property
    def index(self):
        return self.df_tiles.index


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
        img0 = self.get_image(tile_id=0, tile_type=TileType.ANCHOR)
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
                    img_data = self.get_image(tile_id=i, tile_type=tile_type)
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


class MovingWindowImageTilingDataset(ImageSingletDataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        step=(50, 50),
        N_tile=(256, 256),
        rect_indentifier="{scene_id}",
        filter_func=None,
    ):
        """
        Produce moving-window tiling dataset with with step-size defined by
        `step` and tile-size `N_tile` with all scene images found in
        `data_dir`.

        NB: currently all images are loaded into memory when data-loader is
        created. This might not work on machines with little RAM.
        """

        super(ImageSingletDataset).__init__()

        def get_rect_info(fpath):
            img = Image.open(Path(data_dir) / fpath)
            nx, ny = img.size
            return dict(nx=nx, ny=ny)

        self.df_rect_images = _find_tile_files(
            data_dir,
            tile_identifier_format=rect_indentifier,
            aux_accessor=get_rect_info,
        ).set_index("scene_id")
        self.transform = transform

        if filter_func is not None:
            self.df_rect_images = filter_func(df_rect_images=self.df_rect_images)

        # for now we require that all rect images have the same shape
        nxs = set(self.df_rect_images["nx"].values)
        nys = set(self.df_rect_images["ny"].values)
        assert len(nxs) == len(nys) == 1
        self.nx, self.ny = list(nxs)[0], list(nys)[0]

        self.nxt, self.nyt = N_tile
        self.x_step, self.y_step = step

        # the starting image x-index for tile with tile x-index `i`
        self.img_idx_tile_i = np.arange(0, self.nx - self.nxt + 1, self.x_step)
        # the starting image y-index for tile with tile y-index `j`
        self.img_idx_tile_j = np.arange(0, self.ny - self.nyt + 1, self.y_step)

        # number of tiles in x- and y-direction
        self.nt_x = len(self.img_idx_tile_i)
        self.nt_y = len(self.img_idx_tile_j)

        self.df_tiles = self._generate_tiles_dataframe()

        image_load_transforms = get_load_transforms()
        self.img_data_normed = {}
        self.img = {}
        for index in self.df_rect_images.index:
            fpath = self.df_rect_images.loc[index].filepath
            self.img[index] = Image.open(Path(data_dir) / fpath)
            self.img_data_normed[index] = image_load_transforms(self.img[index])

    @property
    def n_scenes(self):
        return len(self.df_rect_images)

    def spatial_index_to_img_ij(self, spatial_index):
        """
        Turn the tile_id (running from 0 to the number of tiles) into the
        (i,j)-index for the tile
        """
        # tile-index in (i,j) running for the number of tiles in each direction
        j = spatial_index // self.nt_x
        i = spatial_index - self.nt_x * j

        # indecies into tile shape
        i_img = self.img_idx_tile_i[i]
        j_img = self.img_idx_tile_j[j]
        return i_img, j_img

    def index_to_spatial_and_scene_index(self, index):
        ntiles_per_scene = self.nt_x * self.nt_y
        scene_index = index // ntiles_per_scene
        spatial_index = index % ntiles_per_scene

        return self.df_rect_images.index[scene_index], spatial_index

    def get_image(self, tile_id):
        scene_index, spatial_index = self.index_to_spatial_and_scene_index(
            index=tile_id
        )

        x_slice, y_slice = self._get_image_tiling_slices(spatial_index=spatial_index)
        # for a PIL image we need the unnormalized image data
        img_data = np.array(self.img[scene_index])
        # PIL images have shape [height, width, #channels]
        img_data_tile = img_data[y_slice, x_slice, :]
        return Image.fromarray(img_data_tile)

    def _get_image_tiling_slices(self, spatial_index):
        i_img, j_img = self.spatial_index_to_img_ij(spatial_index=spatial_index)

        x_slice = slice(i_img, i_img + self.nxt)
        y_slice = slice(j_img, j_img + self.nyt)
        return (x_slice, y_slice)

    def __len__(self):
        return len(self.df_tiles)

    def __getitem__(self, index):
        scene_index, spatial_index = self.index_to_spatial_and_scene_index(index=index)

        x_slice, y_slice = self._get_image_tiling_slices(spatial_index=spatial_index)
        img_data_tile = self.img_data_normed[scene_index][:, y_slice, x_slice]

        if self.transform is not None:
            img_data_tile = self.transform(img_data_tile)
        return img_data_tile

    def add_tiling_coords_to_embedding_dataarray(self, da_emb):
        """
        add coordinates to embedding xr.DataArray so that we can "unstack"
        the 2D array with coords (tile_id, emb_dim) to have coords (scene_id,
        i0, j0, emb_dim), where `i0` and `j0` represent the index of the
        pixel in the original image at the center of each tile

        To unstack, first set the index `tile_id` to use the `scene_id`, `i0`
        and `j0` as a MultiIndex and then unstack `tile_id`
        >>> da_emb.set_index(tile_id=("scene_id", "i0", "j0")).unstack("tile_id")
        """
        scene_id, spatial_index = self.index_to_spatial_and_scene_index(
            da_emb.tile_id.values
        )

        i_img_tile, j_img_tile = self.spatial_index_to_img_ij(
            spatial_index=spatial_index
        )
        i_img_tile_center = (i_img_tile + 0.5 * self.nxt).astype(int)
        j_img_tile_center = (j_img_tile + 0.5 * self.nyt).astype(int)
        da_emb_copy = da_emb.copy()
        da_emb_copy["i0"] = ("tile_id"), i_img_tile_center
        da_emb_copy["j0"] = ("tile_id"), j_img_tile_center

        da_emb_copy["scene_id"] = ("tile_id"), scene_id

        da_emb_copy.attrs["tile_nx"] = self.nxt
        da_emb_copy.attrs["tile_ny"] = self.nyt
        return da_emb_copy

    @property
    def index(self):
        # TODO: shouldn't recreate this on every call, maybe we should create a
        # dataframe for all the tiles
        return np.arange(len(self))

    def _generate_tiles_dataframe(self):
        num_items = self.nt_x * self.nt_y * self.n_scenes

        tile_ids = np.arange(num_items)
        scene_id, spatial_index = self.index_to_spatial_and_scene_index(tile_ids)
        ij_index = self.spatial_index_to_img_ij(spatial_index)

        ds_tiles = xr.Dataset(coords=dict(tile_id=tile_ids))
        ds_tiles["scene_id"] = ("tile_id",), scene_id
        ds_tiles["scene_filepath"] = ("tile_id",), self.df_rect_images.loc[
            scene_id
        ].filepath
        ds_tiles["i"] = ("tile_id",), ij_index[0]
        ds_tiles["j"] = ("tile_id",), ij_index[1]
        return ds_tiles.to_dataframe()
