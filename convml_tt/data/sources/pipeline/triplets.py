from pathlib import Path

import luigi
import numpy as np
import regridcart as rc

from ....pipeline import XArrayTarget, YAMLTarget
from ...common import TILE_IDENTIFIER_FORMAT
from .. import DataSource, goes16
from ..sampling import domain as sampling_domain
from ..sampling import triplets as triplet_sampling
from ..utils.domain_images import rgb_image_from_scene_data
from . import GenerateSceneIDs
from .sampling import (CropSceneSourceFiles, SceneSourceFiles,
                       _SceneRectSampleBase)


class TripletSceneSplits(luigi.Task):
    """
    Work out which scenes all triplets should be sampled from
    """

    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)

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

            for i, (collection_name, N_triplets_collection) in enumerate(
                N_triplets.items()
            ):
                if i <= N_scenes_total - 1:
                    f = N_triplets_collection / N_triplets_total
                    N_scenes_collection = int(f * N_scenes_total)
                else:
                    N_scenes_collection = len(scene_ids_shuffled)

                collection_scene_ids, scene_ids_shuffled = split_list(
                    scene_ids_shuffled, N_scenes_collection
                )
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
                "under `sampling` for the dataset meta info. At minimum "
                "it should contain the number of triplets (`N_triplets` , "
                "this can be a dictionary to have multiple sets, e.g. "
                "'train', 'study', etc) and the tile size in meters (`tile_size`) "
            )

        triplets_meta = ds.sampling["triplets"]
        N_triplets = triplets_meta["N_triplets"]
        if type(N_triplets) == int:
            N_triplets = dict(train=N_triplets)
        scene_collections_splitting = triplets_meta["scene_collections_splitting"]

        if not len(scene_ids) >= 2:
            raise Exception(
                "At least 2 scenes are needed to do `random_by_relative_sample_size`."
                " Please increase the number of scenes in your data source"
            )

        scene_ids_by_collection = self._split_scene_ids(
            scene_ids=scene_ids,
            method=scene_collections_splitting,
            N_triplets=N_triplets,
        )

        tiles_per_scene = {}

        for triplet_collection, n_triplets in N_triplets.items():
            collection_scene_ids = scene_ids_by_collection[triplet_collection]
            for n in range(n_triplets):
                # pick two random scene IDs, ensuring that they are different
                scene_id_anchor, scene_id_distant = np.random.choice(
                    collection_scene_ids, size=2, replace=False
                )

                scene_ids = [scene_id_anchor, scene_id_distant]

                for scene_id, is_distant in zip(scene_ids, [False, True]):
                    scene_tiles = tiles_per_scene.setdefault(str(scene_id), [])
                    scene_tiles.append(
                        dict(
                            triplet_id=n,
                            is_distant=is_distant,
                            triplet_collection=triplet_collection,
                        )
                    )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tiles_per_scene)

    def output(self):
        fn = "tiles_per_scene.yaml"
        p = Path(self.data_path) / "triplets" / fn
        return YAMLTarget(str(p))


class SceneTileLocations(luigi.Task):
    """
    For a given scene work out the sampling locations of all the tiles in it
    """

    data_path = luigi.Parameter(default=".")
    scene_id = luigi.Parameter()

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return dict(
            scene_splits=TripletSceneSplits(data_path=self.data_path),
            scene_source_data=SceneSourceFiles(
                scene_id=self.scene_id, data_path=self.data_path
            ),
        )

    def run(self):
        tiles_per_scene = self.input()["scene_splits"].open()

        tile_locations = []
        if self.scene_id not in tiles_per_scene:
            # we will write an empty file since we don't need to sample tiles
            # from this scene
            pass
        else:
            tiles_meta = tiles_per_scene[self.scene_id]

            triplets_meta = self.data_source.sampling["triplets"]
            neigh_dist_scaling = triplets_meta.get("neigh_dist_scaling", 1.0)
            tile_size = triplets_meta["tile_size"]

            domain = self.data_source.domain
            if isinstance(domain, sampling_domain.SourceDataDomain):
                ds_scene = self.input()["scene_source_data"].open()
                domain = domain.generate_from_dataset(ds=ds_scene)

            for tile_meta in tiles_meta:
                triplet_tile_locations = triplet_sampling.generate_triplet_location(
                    domain=domain,
                    tile_size=tile_size,
                    neigh_dist_scaling=neigh_dist_scaling,
                )

                tile_types = []
                if tile_meta["is_distant"]:
                    tile_types.append("distant")
                    triplet_tile_locations = triplet_tile_locations[-1:]
                else:
                    tile_types.append("anchor")
                    tile_types.append("neighbor")
                    triplet_tile_locations = triplet_tile_locations[:-1]

                for (tile_type, tile_domain) in zip(tile_types, triplet_tile_locations):
                    tile_meta = dict(
                        loc=tile_domain.serialize(),
                        tile_type=tile_type,
                        triplet_id=tile_meta["triplet_id"],
                        triplet_collection=tile_meta["triplet_collection"],
                    )
                    tile_locations.append(tile_meta)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tile_locations)

    def output(self):
        fn = f"tile_locations.{self.scene_id}.yaml"
        p = Path(self.data_path) / "triplets" / fn
        return YAMLTarget(str(p))


class SceneTilesData(_SceneRectSampleBase):
    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        data_source = self.data_source

        reqs = {}
        if isinstance(data_source.domain, sampling_domain.SourceDataDomain):
            reqs["source_data"] = SceneSourceFiles(
                scene_id=self.scene_id,
                data_path=self.data_path,
            )
        else:
            reqs["source_data"] = CropSceneSourceFiles(
                scene_id=self.scene_id,
                data_path=self.data_path,
                pad_ptc=self.crop_pad_ptc,
            )

        reqs["tile_locations"] = SceneTileLocations(
            data_path=self.data_path, scene_id=self.scene_id
        )

        return reqs

    def run(self):
        inputs = self.input()
        source_data_input = inputs["source_data"]
        # for cropped fields the parent task returns a dictionary so that
        # we can have the rendered image too (if that has been produced)
        if isinstance(source_data_input, dict):
            da_src = source_data_input["data"].open()
        else:
            da_src = source_data_input.open()

        domain = self.data_source.domain
        if isinstance(domain, sampling_domain.SourceDataDomain):
            domain = domain.generate_from_dataset(ds=da_src)

        data_source = self.data_source
        dx = data_source.sampling["resolution"]

        for tile_identifier, tile_domain in self.tile_domains:
            da_tile = rc.resample(domain=tile_domain, da=da_src, dx=dx)
            tile_output = self.output()[tile_identifier]
            tile_output["data"].write(da_tile)

            img_tile = rgb_image_from_scene_data(
                data_source=data_source, da_scene=da_tile, src_attrs=da_src.attrs
            )
            img_tile.save(str(tile_output["image"].fn))

    @property
    def tile_domains(self):
        tiles_meta = self.input()["tile_locations"].open()

        for tile_meta in tiles_meta:
            tile_domain = rc.deserialise_domain(tile_meta["loc"])
            tile_identifier = TILE_IDENTIFIER_FORMAT.format(**tile_meta)

            yield tile_identifier, tile_domain

    def output(self):
        if not self.input()["tile_locations"].exists():
            return luigi.LocalTarget("__fakefile__.nc")

        tiles_meta = self.input()["tile_locations"].open()

        tile_data_path = Path(self.data_path) / "triplets"

        outputs = {}

        for tile_meta in tiles_meta:
            tile_identifier = TILE_IDENTIFIER_FORMAT.format(**tile_meta)
            fn_data = f"{tile_identifier}.nc"
            fn_image = f"{tile_identifier}.png"
            outputs[tile_identifier] = dict(
                data=XArrayTarget(str(tile_data_path / fn_data)),
                image=luigi.LocalTarget(str(tile_data_path / fn_image)),
            )
        return outputs


class GenerateTiles(luigi.Task):
    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)

    def run(self):
        scene_ids = list(self.input().open().keys())

        tasks_tiles = {}
        for scene_id in scene_ids:
            tasks_tiles[scene_id] = SceneTilesData(scene_id=scene_id)

        yield tasks_tiles

    def output(self):
        fn_output = "tiles_by_scene.yaml"
        p = Path(self.data_path) / "triplets" / fn_output
        return YAMLTarget(str(p))
