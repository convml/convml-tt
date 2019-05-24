from .utils import get_encodings

import fastai
from .architectures.triplet_trainer import NPMultiImageList

from . import interpretation


class TileType:
    """
    Simple enum for mapping into triplet array """
    ANCHOR = 0
    NEIGHBOUR = 1
    DISTANT = 2
