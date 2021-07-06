"""
pipeline tasks related to spatial cropping, reprojection and sampling of scene data
"""
from pathlib import Path
import luigi

from ....pipeline import XArrayTarget
from .. import goes16
from . import AllSceneIDs
from .. import DataSource


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


class CropSceneSourceChannels(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    source_channels = luigi.ListParameter()
    pad_ptc = luigi.Parameter(default=0.1)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        ds = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / ds.source
        tasks = []
        if ds.source == "goes16":
            for source_fn in self.source_channels:
                tasks.append(goes16.pipeline.GOES16Fetch(key=source_fn, data_path=source_data_path))
        else:
            raise NotImplementedError(ds.source)

        return tasks

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

        bbox_domain = ds.domain_bounds

        if "lat" in da_full.coords and "lon" in da_full.coords:
            da_cropped = tiler.crop_field_to_latlon_box(
                da=da_full,
                box=bbox_domain,
                pad_pct=self.pad_ptc,
            )
        else:
            raise NotImplementedError(da_full.coords)

        da_cropped.to_netcdf(self.output().fn)

    def output(self):
        ds = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / ds.source / "cropped"
        fn = "{self.scene_id}__{ds.type}.nc"
        return XArrayTarget(str(source_data_path / fn))
