"""
Utility class for loading tile definitions from existing datasets
"""

import yaml
from tqdm import tqdm

from ..data.sources import satdata
from ..architectures.triplet_trainer import TileType


class TripletTile(satdata.Tile):
    def __init__(self, rgb_img, meta, tile_id, data_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rgb_img = rgb_img
        self.meta = meta
        self.tile_id = tile_id
        self.data_path = data_path

def load_tile_definitions(triplets, tile_type=TileType.ANCHOR):
    def _load_tile_from_triplet(triplet):
        fn_triplet_meta = triplets.TRIPLET_META_FILENAME_FORMAT.format(
            triplet_id=triplet.id
        )
        path_triplet_meta = triplet.src_path/fn_triplet_meta

        # the things stored in the triplet are actually the RGB images, these
        # will be handy for plotting later
        rgb_img = triplet[tile_type]

        meta = yaml.load(open(path_triplet_meta))
        meta_group = meta['target']
        if not tile_type==TileType.ANCHOR:
            raise NotImplementedError

        anchor_meta = meta_group['anchor']
        tile = TripletTile(
            rgb_img=rgb_img,
            meta=dict(rgb_source_files=meta_group['source_files']),
            lat0=anchor_meta['lat'], lon0=anchor_meta['lon'],
            size=anchor_meta['size'],
            tile_id=triplet.id, data_path=triplets.path.absolute()
        )

        return tile

    return [_load_tile_from_triplet(triplet) for triplet in tqdm(triplets)]
