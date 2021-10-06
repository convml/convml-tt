from pathlib import Path

import luigi

from ....pipeline import XArrayTarget, YAMLTarget
from .. import DataSource
from ..sampling import domain as sampling_domain
from ..sampling.interpolation import resample
from . import GenerateSceneIDs
from .sampling import CropSceneSourceFiles, SceneSourceFiles, _SceneRectSampleBase
from ..utils.domain_images import rgb_image_from_scene_data


class SceneRegriddedData(_SceneRectSampleBase):
    """
    Regrid the scene source data to a fixed Cartesian resolution
    """

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

        return reqs

    def run(self):
        inputs = self.input()
        da_src = inputs["source_data"]["data"].open()

        domain = self.data_source.domain
        if isinstance(domain, sampling_domain.SourceDataDomain):
            domain = domain.generate_from_dataset(ds=da_src)

        data_source = self.data_source
        if (
            "rect" not in data_source.sampling
            or data_source.sampling["rect"].get("dx") is None
        ):
            raise Exception(
                "To produce isometric grid resampling of the source data please "
                "define the grid-spacing by setting `dx` in a section called `rect` "
                "in the `sampling` part of the data source meta information"
            )
        dx = data_source.sampling["rect"]["dx"]

        da_domain = resample(domain=domain, da=da_src, dx=dx)
        domain_output = self.output()
        Path(domain_output["data"].fn).parent.mkdir(exist_ok=True, parents=True)
        domain_output["data"].write(da_domain)

        img_domain = rgb_image_from_scene_data(
            data_source=data_source, da_scene=da_domain, src_attrs=da_src.attrs
        )

        img_domain.save(str(domain_output["image"].fn))

    def output(self):
        scene_data_path = Path(self.data_path) / "rect"

        fn_data = f"{self.scene_id}.nc"
        fn_image = f"{self.scene_id}.png"
        return dict(
            data=XArrayTarget(str(scene_data_path / fn_data)),
            image=luigi.LocalTarget(str(scene_data_path / fn_image)),
        )


class GenerateRegriddedScenes(luigi.Task):
    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)

    def run(self):
        scene_ids = list(self.input().open().keys())

        tasks_scenes = {}
        for scene_id in scene_ids:
            tasks_scenes[scene_id] = SceneRegriddedData(scene_id=scene_id)

        yield tasks_scenes

    def output(self):
        fn_output = "regridded_data_by_scene.yaml"
        p = Path(self.data_path) / "rect" / fn_output
        return YAMLTarget(str(p))
