from pathlib import Path
import luigi
import numpy as np

from .sampling import _SceneRectSampleBase, CropSceneSourceChannels
from ...dataset import TILE_FILENAME_FORMAT
from ....pipeline import XArrayTarget, YAMLTarget
from ..sampling import triplets as triplet_sampling
from . import AllSceneIDs
from .. import DataSource


class TripletTileLocations(luigi.Task):
    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return AllSceneIDs(data_path=self.data_path)

    def _split_scene_ids(self, scene_ids, method, N_triplets):
        scene_collections = {}
        if method == "random_by_relative_sample_size":
            # split all scene IDs randomly so that for each collection in
            # `N_triplets` the fraction of scenes allocated equals the fraction
            # of triplets in the collection
            N_scenes_total = len(scene_ids)
            N_triplets_total = sum(N_triplets.values())
            scene_ids_shuffled = np.random.permutation(scene_ids)

            def split_list(arr, idx):
                return arr[:idx], arr[idx:]

            for i, (collection_name, N_triplets_collection) in enumerate(N_triplets.items()):
                if i <= N_scenes_total - 1:
                    f = N_triplets_collection / N_triplets_total
                    N_scenes_collection = int(f * N_scenes_total)
                else:
                    N_scenes_collection = len(scene_ids_shuffled)

                collection_scene_ids, scene_ids_shuffled = split_list(scene_ids_shuffled, N_scenes_collection)
                scene_collections[collection_name] = collection_scene_ids
        else:
            raise NotImplementedError(method)

        return scene_collections

    def run(self):
        scene_ids = list(self.input().open().keys())

        ds = self.data_source

        if "triplets" not in ds.sampling:
            raise Exception(
                "To produce triplets please define a `triplets` section "
                "under `sampling` for the dataset meta info"
            )

        triplets_meta = ds.sampling["triplets"]
        N_triplets = triplets_meta["N_triplets"]
        if type(N_triplets) == int:
            N_triplets = dict(train=N_triplets)
        tile_size = triplets_meta["tile_size"]
        neigh_dist_scaling = triplets_meta.get("neigh_dist_scaling", 1.0)
        scene_collections_splitting = triplets_meta["scene_collections_splitting"]

        scene_ids_by_collection = self._split_scene_ids(
            scene_ids=scene_ids, method=scene_collections_splitting,
            N_triplets=N_triplets
        )

        triplet_locations = {}
        tile_types = ["anchor", "neighbor", "distant"]
        for triplet_collection, n_triplets in N_triplets.items():
            collection_scene_ids = scene_ids_by_collection[triplet_collection]
            triplet_collection_locations = []
            for n in range(n_triplets):
                scene_id_anchor, scene_id_distant = np.random.choice(collection_scene_ids, size=2)
                scene_id_neighbor = scene_id_anchor
                scene_ids_triplet = [scene_id_anchor, scene_id_neighbor, scene_id_distant]

                triplet_location = triplet_sampling.generate_triplet_location(
                    domain=ds.domain, tile_size=tile_size, neigh_dist_scaling=neigh_dist_scaling,
                )

                triplet_meta = dict()
                for (tile_type, scene_id, tile_domain) in zip(tile_types, scene_ids_triplet, triplet_location):
                    triplet_meta[tile_type] = dict(
                        id=n,
                        scene_id=str(scene_id),
                        loc=tile_domain.serialize()
                    )

                triplet_collection_locations.append(triplet_meta)
            triplet_locations[triplet_collection] = triplet_collection_locations

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(triplet_locations)

    def output(self):
        fn = "triplet_locations.yaml"
        p = Path(self.data_path) / "triplets" / fn
        return YAMLTarget(str(p))


class TripletTileData(_SceneRectSampleBase):
    tile_type = luigi.Parameter()
    triplet_id = luigi.Parameter()
    triplet_collection_id = luigi.Parameter()

    def requires(self):
        reqs = {}
        reqs["source_data"] = CropSceneSourceChannels(
            scene_id=self.scene_id,
            data_path=self.data_path,
            source_channels=self.source_channels,
            pad_ptc=self.crop_pad_ptc,
        )

        reqs["triplet_locations"] = TripletTileLocations(
            scene_id=self.scene_id,
            data_path=self.data_path,
        )

        return reqs

    def run(self):
        pass

    def output(self):
        tile_data_path = Path(self.data_path) / "triplets" / self.triplet_collection_id
        tile_id = TILE_FILENAME_FORMAT.format(triplet_id=self.triplet_id, tile_type=self.tile_type)
        fn_data = f"{tile_id}.nc"
        fn_image = f"{tile_id}.png"
        return dict(
            data=XArrayTarget(str(tile_data_path / fn_data)),
            image=XArrayTarget(str(tile_data_path / fn_image)),
        )


class GenerateTriplets(luigi.Task):
    pass
