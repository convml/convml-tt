# coding: utf-8
from . import tiler, FixedTimeRangeSatelliteTripletDataset
from . import satpy_rgb, pipeline as sat_pipeline

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


class XArrayTarget(luigi.target.FileSystemTarget):
    fs = luigi.local_target.LocalFileSystem()

    def __init__(self, path, *args, **kwargs):
        super(XArrayTarget, self).__init__(path, *args, **kwargs)
        self.path = path

    def open(self, *args, **kwargs):
        # ds = xr.open_dataset(self.path, engine='h5netcdf', *args, **kwargs)
        ds = xr.open_dataset(self.path, *args, **kwargs)

        if len(ds.data_vars) == 1:
            name = list(ds.data_vars)[0]
            da = ds[name]
            da.name = name
            return da
        else:
            return ds

    @property
    def fn(self):
        return self.path

class MakeRectRGBDataArray(luigi.Task):
    source_data_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    scene_num = luigi.IntParameter()

    def _get_dataset(self):
        return FixedTimeRangeSatelliteTripletDataset.load(self.dataset_path)

    def requires(self):
        dataset = self._get_dataset()
        if not 'rectpred' in dataset.extra:
            raise Exception("Please define a `rectpred` setup in your"
                            " meta.yaml file")

        source_data_path = Path(self.source_data_path).expanduser()
        scenes_source_fns = dataset.fetch_source_data(
            source_data_path=source_data_path
        )

        source_fns = scenes_source_fns[self.scene_num]
        source_fns = [
            str(self.source_data_path/sat_pipeline.SOURCE_DIR/fn) for fn in source_fns
        ]
        t = sat_pipeline.CreateRGBScene(
            source_fns=source_fns, domain_bbox=dataset.domain_bbox,
            data_path=self.source_data_path,
        )
        return t

    def run(self):
        dataset = self._get_dataset()
        da_scene = self.input().open()
        domain_rect = tiler.RectTile(dataset.extra['rectpred']['domain'])
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
        da_scene = self.input().open()
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
    source_data_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    scene_num = luigi.IntParameter()
    dx = luigi.FloatParameter(default=200.0e3/256)

    def requires(self):
        return MakeRectRGBDataArray(
            source_data_path=self.source_data_path,
            dataset_path=self.dataset_path,
            scene_num=self.scene_num,
            dx=self.dx
        )

    def run(self):
        da_rect = self.input().open()
        img = satpy_rgb.rgb_da_to_img(da_rect)

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        img.save(str(self.output().fn))

    def output(self):
        if not self.input().exists():
            return luigi.LocalTarget('fakefile.png')

        da_scene = self.input().open()
        t = dateutil.parser.parse(da_scene.start_time)
        fn = FILE_FORMAT.format(
            platform_name="CONV-ORG",
            instrument="GOES-R",
            variable="RGB",
            date=t.strftime(DATE_FORMAT),
            time=t.strftime(TIME_FORMAT),
            ext='png'
        )

        p_out = Path(t.strftime(PATH_FORMAT))/fn
        return luigi.LocalTarget(str(p_out))

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
