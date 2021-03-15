# coding: utf-8
from . import tiler, FixedTimeRangeSatelliteTripletDataset
from . import satpy_rgb, pipeline as sat_pipeline
from ....pipeline import XArrayTarget
from ....data.dataset import SceneBulkProcessingBaseTask

from pathlib import Path
import warnings
import luigi


class MakeRectRGBDataArray(luigi.Task):
    dataset_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def _get_dataset(self):
        return FixedTimeRangeSatelliteTripletDataset.load(self.dataset_path)

    def requires(self):
        t = sat_pipeline.CreateRGBScene(
            scene_id=self.scene_id,
            dataset_path=self.dataset_path,
        )

        if t.output().exists():
            try:
                da_scene = t.output().open()
                if da_scene.count() == 0:
                    warnings.warn(
                        "Something is wrong with RGB DataArray file"
                        f" `{t.output().fn}`, it doesn't contain any"
                        "data. Deleting so it can be recreated."
                    )
                    Path(t.output().fn).unlink()
            except Exception:
                print(f"There was a problem opening `{t.output().fn}`")
                raise
        return t

    def run(self):
        da_scene = self.input().open()

        dataset = self._get_dataset()
        domain_rect = dataset.get_domain_rect(da_scene=da_scene)
        da_rect = domain_rect.resample(
            da=da_scene, dx=dataset.extra["rectpred"]["resolution"], keep_attrs=True
        )
        if "crs" in da_rect.attrs:
            del da_rect.attrs["crs"]
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_rect.to_netcdf(self.output().fn)

    def output(self):
        fn = "{}.nc".format(self.scene_id)
        p_out = Path(self.dataset_path) / "composites" / "rect" / fn
        return XArrayTarget(str(p_out))


class MakeRectRGBImage(luigi.Task):
    dataset_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return MakeRectRGBDataArray(
            scene_id=self.scene_id,
            dataset_path=self.dataset_path,
        )

    def run(self):
        da_rect = self.input().open()
        img = satpy_rgb.rgb_da_to_img(da_rect)

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        img.save(str(self.output().fn))

    def output(self):
        fn = "{}.png".format(self.scene_id)
        p_out = Path(self.dataset_path) / "composites" / "rect" / fn
        return luigi.LocalTarget(str(p_out))


class MakeAllRectRGBDataArrays(SceneBulkProcessingBaseTask):
    TaskClass = MakeRectRGBImage

    def _get_task_class_kwargs(self):
        return {}
