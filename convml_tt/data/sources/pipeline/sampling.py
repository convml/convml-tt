"""
pipeline tasks related to spatial cropping, reprojection and sampling of scene data
"""
from pathlib import Path
import luigi

from ....pipeline import XArrayTarget, ImageTarget
from .. import goes16
from . import AllSceneIDs
from .. import DataSource
from ..sampling.cropping import crop_field_to_domain


class SceneSourceFiles(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return AllSceneIDs(data_path=self.data_path)

    def _build_fetch_tasks(self):
        task = None
        ds = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / ds.source

        if self.input().exists():
            all_source_files = self.input().open()
            scene_source_files = all_source_files[self.scene_id]
            if ds.source == "goes16":
                task = goes16.pipeline.GOES16Fetch(keys=scene_source_files, data_path=source_data_path)
            else:
                raise NotImplementedError(ds.source)

        return task

    def run(self):
        yield self._build_fetch_tasks()

    def output(self):
        source_task = self._build_fetch_tasks()
        if source_task is not None:
            return source_task.output()
        else:
            return luigi.LocalTarget(f"__fake_target__{self.scene_id}__")


class CropSceneSourceFiles(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    pad_ptc = luigi.Parameter(default=0.1)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return SceneSourceFiles(data_path=self.data_path, scene_id=self.scene_id)

    def run(self):
        ds = self.data_source
        inputs = self.input()

        if ds.source == "goes16":
            if ds.type == "truecolor_rgb":
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
                raise NotImplementedError(ds.type)
        else:
            raise NotImplementedError(ds.source)

        da_cropped = crop_field_to_domain(domain=ds.domain, da=da_full, pad_pct=self.pad_ptc)

        if ds.source == "goes16" and ds.type == "truecolor_rgb":
            img_cropped = goes16.satpy_rgb.rgb_da_to_img(da=da_cropped)
        else:
            raise NotImplementedError(ds.source)

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
    crop_pad_ptc = luigi.Parameter(default=0.1)

    def requires(self):
        t_scene_ids = AllSceneIDs(data_path=self.data_path)
        if not t_scene_ids.output().exists():
            raise Exception(
                "Scene IDs haven't been defined for this dataset yet "
                "Please run the `AllSceneIDs task first`"
            )
        all_scenes = t_scene_ids.output().open()
        source_channels = all_scenes[self.scene_id]

        return CropSceneSourceChannels(
            scene_id=self.scene_id,
            data_path=self.data_path,
            source_channels=source_channels,
            pad_ptc=self.crop_pad_ptc
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
