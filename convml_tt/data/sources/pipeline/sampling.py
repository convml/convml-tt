"""
pipeline tasks related to spatial cropping, reprojection and sampling of scene data
"""
from pathlib import Path
import luigi

from ....pipeline import XArrayTarget, ImageTarget
from .. import goes16
from . import GenerateSceneIDs
from .. import DataSource
from ..sampling.cropping import crop_field_to_domain
from ..sampling.domain import LocalCartesianDomain
from ..les import LESDataFile


class SceneSourceFiles(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)

    def _build_fetch_tasks(self):
        task = None
        ds = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / ds.source

        if self.input().exists():
            all_source_files = self.input().open()
            scene_source_files = all_source_files[self.scene_id]
            if ds.source == "goes16":
                task = goes16.pipeline.GOES16Fetch(
                    keys=scene_source_files, data_path=source_data_path
                )
            elif ds.source == "LES":
                # assume that these files already exist
                task = LESDataFile(file_path=scene_source_files)
            else:
                raise NotImplementedError(ds.source)

        return task

    def run(self):
        fetch_tasks = self._build_fetch_tasks()
        if fetch_tasks is not None:
            yield fetch_tasks

    def output(self):
        source_task = self._build_fetch_tasks()
        if source_task is not None:
            return source_task.output()
        else:
            return luigi.LocalTarget(f"__fake_target__{self.scene_id}__")


class CropSceneSourceFiles(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    pad_ptc = luigi.FloatParameter(default=0.1)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return SceneSourceFiles(data_path=self.data_path, scene_id=self.scene_id)

    def run(self):
        data_source = self.data_source

        if data_source.source == "goes16":
            inputs = self.input()
            if data_source.type == "truecolor_rgb":
                if not len(inputs) == 3:
                    raise Exception(
                        "To create TrueColor RGB images for GOES-16 the first"
                        " three Radiance channels (1, 2, 3) are needed"
                    )

                scene_fns = [inp.fn for inp in inputs]
                da_full = goes16.satpy_rgb.load_rgb_files_and_get_composite_da(
                    scene_fns=scene_fns
                )
            else:
                raise NotImplementedError(data_source.type)
        elif data_source.source == "LES":
            domain = data_source.domain
            ds_input = self.input().open()
            if isinstance(domain, LocalCartesianDomain):
                domain.validate_dataset(ds=ds_input)

            raise NotADirectoryError(42)
        else:
            raise NotImplementedError(data_source.source)

        # import ipdb
        # with ipdb.launch_ipdb_on_exception():
        da_cropped = crop_field_to_domain(
            domain=data_source.domain, da=da_full, pad_pct=self.pad_ptc
        )

        if data_source.source == "goes16" and data_source.type == "truecolor_rgb":
            img_cropped = goes16.satpy_rgb.rgb_da_to_img(da=da_cropped)
            del da_cropped.attrs["_satpy_id"]
        else:
            raise NotImplementedError(data_source.source)

        self.output_path.mkdir(exist_ok=True, parents=True)
        self.output()["data"].write(da_cropped)
        self.output()["image"].write(img_cropped)

    @property
    def output_path(self):
        ds = self.data_source
        return Path(self.data_path) / "source_data" / ds.source / ds.type / "cropped"

    def output(self):
        data_path = self.output_path
        fn_data = f"{self.scene_id}.nc"
        fn_image = f"{self.scene_id}.png"
        return dict(
            data=XArrayTarget(str(data_path / fn_data)),
            image=ImageTarget(str(data_path / fn_image)),
        )


class _SceneRectSampleBase(luigi.Task):
    """
    This task represents the creation of scene data to feed into a neural
    network, either the data for an entire rectangular domain or a single tile
    """

    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    crop_pad_ptc = luigi.FloatParameter(default=0.1)

    def requires(self):
        t_scene_ids = GenerateSceneIDs(data_path=self.data_path)
        if not t_scene_ids.output().exists():
            raise Exception(
                "Scene IDs haven't been defined for this dataset yet "
                "Please run the `GenerateSceneIDs task first`"
            )
        # all_scenes = t_scene_ids.output().open()
        # source_channels = all_scenes[self.scene_id]

        return CropSceneSourceFiles(
            scene_id=self.scene_id, data_path=self.data_path, pad_ptc=self.crop_pad_ptc
        )


class SceneRectData(_SceneRectSampleBase):
    def output(self):
        scene_data_path = Path(self.data_path) / "scenes"
        fn_data = f"{self.scene_id}.nc"
        fn_image = f"{self.scene_id}.png"
        return dict(
            data=XArrayTarget(str(scene_data_path / fn_data)),
            image=XArrayTarget(str(scene_data_path / fn_image)),
        )