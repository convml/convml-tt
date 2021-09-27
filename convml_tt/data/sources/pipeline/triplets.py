from pathlib import Path
import luigi
import numpy as np
from PIL import Image

from .sampling import _SceneRectSampleBase, CropSceneSourceFiles, SceneSourceFiles
from ...dataset import TILE_IDENTIFIER_FORMAT
from ....pipeline import XArrayTarget, YAMLTarget
from ..sampling import triplets as triplet_sampling, domain as sampling_domain
from ..sampling.interpolation import resample
from . import GenerateSceneIDs
from .. import DataSource


class TripletTileLocations(luigi.Task):
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
            scene_ids=scene_ids,
            method=scene_collections_splitting,
            N_triplets=N_triplets,
        )

        tile_locations = []

        for triplet_collection, n_triplets in N_triplets.items():
            collection_scene_ids = scene_ids_by_collection[triplet_collection]
            for n in range(n_triplets):
                tile_types = ["anchor", "neighbor", "distant"]

                scene_id_anchor, scene_id_distant = np.random.choice(
                    collection_scene_ids, size=2
                )
                scene_id_neighbor = scene_id_anchor
                scene_ids_triplet = [
                    scene_id_anchor,
                    scene_id_neighbor,
                    scene_id_distant,
                ]

                domain = self.data_source.domain
                if isinstance(domain, sampling_domain.SourceDataDomain):
                    t_anchor_output = yield SceneSourceFiles(
                        scene_id=scene_id_anchor, data_path=self.data_path
                    )
                    ds_anchor = t_anchor_output.open()
                    domain = domain.generate_from_dataset(ds=ds_anchor)

                triplet_location = triplet_sampling.generate_triplet_location(
                    domain=domain,
                    tile_size=tile_size,
                    neigh_dist_scaling=neigh_dist_scaling,
                )

                for (tile_type, scene_id, tile_domain) in zip(
                    tile_types, scene_ids_triplet, triplet_location
                ):
                    tile_meta = dict(
                        scene_id=str(scene_id),
                        loc=tile_domain.serialize(),
                        tile_type=tile_type,
                        triplet_id=n,
                        triplet_collection=triplet_collection,
                    )
                    tile_locations.append(tile_meta)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tile_locations)

    def output(self):
        fn = "triplet_locations.yaml"
        p = Path(self.data_path) / "triplets" / fn
        return YAMLTarget(str(p))


class SceneTripletsTileData(_SceneRectSampleBase):
    tiles_meta = luigi.DictParameter()

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

        reqs["triplet_locations"] = TripletTileLocations(
            data_path=self.data_path,
        )

        return reqs

    def run(self):
        inputs = self.input()
        da_src = inputs["source_data"].open()

        domain = self.data_source.domain
        if isinstance(domain, sampling_domain.SourceDataDomain):
            domain = domain.generate_from_dataset(ds=da_src)

        triplets_meta = self.data_source.sampling["triplets"]
        tile_size = triplets_meta["tile_size"]
        tile_N = triplets_meta["tile_N"]
        dx = tile_size / tile_N

        for tile_meta in self.tiles_meta:
            tile_identifier = TILE_IDENTIFIER_FORMAT.format(**tile_meta)

            tile_domain = sampling_domain.deserialise_domain(tile_meta["loc"])
            da_tile = resample(domain=tile_domain, da=da_src, dx=dx)
            tile_output = self.output()[tile_identifier]
            tile_output["data"].write(da_tile)

            # TODO: make a more generic image generation function
            img_data = da_tile.data
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            img_data = (img_data * 255).astype(np.uint8)
            tile_img = Image.fromarray(img_data)
            tile_img.save(str(tile_output["image"].fn))

    def output(self):
        tile_data_path = Path(self.data_path) / "triplets"

        outputs = {}

        for tile_meta in self.tiles_meta:
            tile_identifier = TILE_IDENTIFIER_FORMAT.format(**tile_meta)
            fn_data = f"{tile_identifier}.nc"
            fn_image = f"{tile_identifier}.png"
            outputs[tile_identifier] = dict(
                data=XArrayTarget(str(tile_data_path / fn_data)),
                image=luigi.LocalTarget(str(tile_data_path / fn_image)),
            )
        return outputs


class GenerateTriplets(luigi.Task):
    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        reqs = {}
        reqs["tile_locations"] = TripletTileLocations(
            data_path=self.data_path,
        )

        return reqs

    def run(self):
        inputs = self.input()
        tile_locs = inputs["tile_locations"].open()

        tile_locs_by_scene = {}
        for tile_meta in tile_locs:
            scene_id = tile_meta.pop("scene_id")
            tile_locs_by_scene.setdefault(scene_id, []).append(tile_meta)

        tasks_tiles = {}
        for scene_id, tiles_meta in tile_locs_by_scene.items():
            tasks_tiles[scene_id] = SceneTripletsTileData(
                scene_id=scene_id, tiles_meta=tiles_meta
            )

        yield tasks_tiles

        self.output().write(tile_locs_by_scene)

    def output(self):
        fn_output = "tiles_by_scene.yaml"
        p = Path(self.data_path) / "triplets" / fn_output
        return YAMLTarget(str(p))
