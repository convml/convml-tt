# coding: utf-8
from . import tiler, FixedTimeRangeSatelliteTripletDataset
from . import satpy_rgb, pipeline as sat_pipeline
from ....pipeline import XArrayTarget, YAMLTarget

from pathlib import Path
import dateutil
import luigi

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

# PlatformName_Instrument_Variable_YYYYMMDD_HHmmSS_SpaceCov_SpaceRes_Level_ForcastNB.ext
FILE_FORMAT = "{platform_name}_{instrument}_{variable}_{date}_{time}.{ext}"
DATE_FORMAT = "%Y%m%d"
TIME_FORMAT = "%H%M%S"

PATH_FORMAT = "%Y/%Y_%m_%d"


class MakeRectRGBDataArray(luigi.Task):
    dataset_path = luigi.Parameter()
    scene_path = luigi.Parameter()

    def _get_dataset(self):
        return FixedTimeRangeSatelliteTripletDataset.load(self.dataset_path)

    def requires(self):
        dataset = self._get_dataset()
        if not 'rectpred' in dataset.extra:
            raise Exception("Please define a `rectpred` setup in your"
                            " meta.yaml file")

        return dataset.fetch_source_data()

    def run(self):
        # need to use this to include the projection meta information
        da_scene = sat_pipeline.RGBCompositeNetCDFFile(self.scene_path).open()

        dataset = self._get_dataset()
        domain_rect = tiler.RectTile(**dataset.extra['rectpred']['domain'])
        da_rect = domain_rect.resample(
            da=da_scene,
            dx=dataset.extra['rectpred']['resolution'],
            keep_attrs=True
        )
        if 'crs' in da_rect.attrs:
            del(da_rect.attrs['crs'])
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_rect.to_netcdf(self.output().fn)

    def output(self):
        da_scene = xr.open_dataarray(self.scene_path)
        t = dateutil.parser.parse(da_scene.start_time)
        fn = FILE_FORMAT.format(
            platform_name="CONV-ORG",
            instrument="GOES-R",
            variable="RGB",
            date=t.strftime(DATE_FORMAT),
            time=t.strftime(TIME_FORMAT),
            ext='nc'
        )

        p_out = Path(t.strftime(PATH_FORMAT))/fn
        return XArrayTarget(str(p_out))

class MakeRectRGBImage(luigi.Task):
    dataset_path = luigi.Parameter()
    scene_path = luigi.Parameter()

    def requires(self):
        return MakeRectRGBDataArray(
            scene_path=self.scene_path,
            dataset_path=self.dataset_path,
        )

    def run(self):
        da_rect = self.input().open()
        img = satpy_rgb.rgb_da_to_img(da_rect)

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        img.save(str(self.output().fn))

    def output(self):
        if not Path(self.scene_path).exists():
            return None

        da_scene = xr.open_dataarray(self.scene_path)
        t = dateutil.parser.parse(da_scene.start_time)
        fn = FILE_FORMAT.format(
            platform_name="CONV-ORG",
            instrument="GOES-R",
            variable="RGB",
            date=t.strftime(DATE_FORMAT),
            time=t.strftime(TIME_FORMAT),
            ext='png'
        )

        p_out = Path(self.dataset_path)/"composites"/"rect"/t.strftime(PATH_FORMAT)/fn
        return luigi.LocalTarget(str(p_out))

class MakeAllRectRGBDataArrays(luigi.Task):
    dataset_path = luigi.Parameter()

    def _get_dataset(self):
        return FixedTimeRangeSatelliteTripletDataset.load(self.dataset_path)

    def requires(self):
        dataset = self._get_dataset()
        if not 'rectpred' in dataset.extra:
            raise Exception("Please define a `rectpred` setup in your"
                            " meta.yaml file")

        return dataset.fetch_source_data()

    def run(self):
        dataset = self._get_dataset()
        scene_source_fns = self.input().read()

        scene_tasks = []
        for source_fns in scene_source_fns:
            # first we need to create RGB scene file, this will be in the
            # original resolution of the input
            t_scene = sat_pipeline.CreateRGBScene(
                source_fns=source_fns,
                domain_bbox=dataset.domain_bbox,
                data_path=self.dataset_path
            )
            scene_tasks.append(t_scene)
        scene_outputs = yield scene_tasks

        image_tasks = []
        for t_output in scene_outputs:
            # and then we create the resampled image (or just the source array)
            t = MakeRectRGBImage(
                dataset_path=self.dataset_path,
                scene_path=t_output.fn
            )
            image_tasks.append(t)
        image_outputs = yield image_tasks

        self.output().write([
            [scene_output.fn, image_output.fn]
            for (scene_output, image_output)
            in zip(scene_outputs, image_outputs)
        ])

    def output(self):
        fn = "all_scenes.yaml"
        p = Path(self.dataset_path)/"composites"/"rect"/fn
        return YAMLTarget(str(p))

def _plot_scene(da_scene, dataset):
    fig, ax = plt.subplots(subplot_kw=dict(projection=da_scene.crs), figsize=(12,6))
    da_scene.coarsen(x=20, y=20, boundary="trim").mean().sel(bands="B").plot(transform=da_scene.crs, ax=ax)
    ax.gridlines()
    ax.coastlines()

    rect.plot_outline(ax=ax)
    dataset.get_domain().plot_outline(ax=ax, color='red', alpha=0.3)

def _plot_rect_grid(rect):
    ax = rect.plot_outline()
    grid = rect.get_grid(dx=100e3)
    ax.scatter(grid.lon, grid.lat, transform=ccrs.PlateCarree())
    plt.margins(0.5)
