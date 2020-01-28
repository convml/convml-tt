# coding: utf-8

from convml_tt.data.sources.satellite import tiler, FixedTimeRangeSatelliteTripletDataset
import convml_tt.data.sources.satellite.satpy_rgb
import convml_tt.data.sources.satellite.pipeline as sat_pipeline

from pathlib import Path
import dateutil
import luigi

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

# PlatformName_Instrument_Variable_YYYYMMDD_HHmmSS_SpaceCov_SpaceRes_Level_ForcastNB.ext
FILE_FORMAT = "{platform_name}_{instrument}_{variable}_{date}_{time}.png"
DATE_FORMAT = "%Y%m%d"
TIME_FORMAT = "%H%M%S"

PATH_FORMAT = "%Y/%Y_%m_%d"

DX = 200e3/256

DOMAIN_RECT = tiler.RectTile(lat0=14, lon0=-48, l_zonal=3000e3, l_meridional=1000e3)

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

class MakeRectRGBImage(luigi.Task):
    source_data_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    scene_num = luigi.IntParameter()

    def requires(self):
        dataset = FixedTimeRangeSatelliteTripletDataset.load(self.dataset_path)

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
        self.output().fn.parent.mkdir(exist_ok=True, parents=True)
        da_scene = self.input().open()
        img = DOMAIN_RECT.create_true_color_img(da_scene, resampling_dx=DX)
        img_pil = img.pil_image()
        img_pil.save(str(self.output().fn))

    def output(self):
        da_scene = self.input().open()
        t = dateutil.parser.parse(da_scene.start_time)
        fn = FILE_FORMAT.format(
            platform_name="CONV-ORG",
            instrument="GOES-R",
            variable="RGB",
            date=t.strftime(DATE_FORMAT),
            time=t.strftime(TIME_FORMAT),
        )

        p_out = Path(t.strftime(PATH_FORMAT))/fn
        return luigi.LocalTarget(p_out)
