import numpy as np


from ...data.dataset import ImageSingletDataset, get_load_transforms
from ...utils import get_embeddings
from ...data.transforms import get_transforms


class MovingWindowImageTilingDataset(ImageSingletDataset):
    def __init__(self, img, step, N_tile, transform):
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
        self.img_data = image_load_transforms(self.img)
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

    def __getitem__(self, index):
        i_img, j_img = self.index_to_img_ij(index=index)

        x_slice = slice(i_img, i_img + self.nxt)
        y_slice = slice(j_img, j_img + self.nyt)
        img_data_tile = self.img_data[:, y_slice, x_slice]

        return self.transform(img_data_tile)


def make_sliding_tile_model_predictions(
    img, model, step=(50, 50), prediction_batch_size=32
):
    """
    Produce moving-window prediction array from image `img` with `model` with
    step-size defined by `step` and tile-size `N_tile`.

    NB: j-indexing is from "top_left" i.e. is likely in the opposite order to
    what would be expected for y-axis of original image (positive being up)

    i0 and j0 coordinates denote center of each prediction tile
    """
    # TODO: make `N_tile` a parameter of the model the dataset is used with
    N_tile = (256, 256)

    # need to ensure the image-tiles that are sampled are transformed to match
    # the base-architecture (in case we're using a pretrained network)
    transform = get_transforms(step="predict", normalize_for_arch=model.base_arch)
    dataset = MovingWindowImageTilingDataset(
        img=img, step=step, N_tile=N_tile, transform=transform
    )

    da_emb = get_embeddings(
        tile_dataset=dataset, model=model, prediction_batch_size=prediction_batch_size
    )

    # "unstack" the 2D array with coords (tile_id, emb_dim) to have coords (i0,
    # j0, emb_dim), where `i0` and `j0` represent the index of the pixel in the
    # original image at the center of each tile
    i_img_tile, j_img_tile = dataset.index_to_img_ij(da_emb.tile_id.values)
    i_img_tile_center = (i_img_tile + 0.5 * dataset.nxt).astype(int)
    j_img_tile_center = (j_img_tile + 0.5 * dataset.nyt).astype(int)
    da_emb["i0"] = ("tile_id"), i_img_tile_center
    da_emb["j0"] = ("tile_id"), j_img_tile_center
    da_emb = da_emb.set_index(tile_id=("i0", "j0")).unstack("tile_id")

    da_emb.attrs["tile_nx"] = dataset.nxt
    da_emb.attrs["tile_ny"] = dataset.nyt

    return da_emb
