import luigi
import xarray as xr
import yaml
from pathlib import Path
from PIL import Image

import coloredlogs

coloredlogs.install()


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

    def write(self, ds):
        ds.to_netcdf(self.fn)


class YAMLTarget(luigi.LocalTarget):
    def write(self, obj):
        with super().open("w") as fh:
            yaml.dump(obj, fh)

    def read(self):
        with super().open() as fh:
            return yaml.load(fh, Loader=yaml.FullLoader)

    def open(self):
        return self.read()

    def exists(self):
        return Path(self.path).exists()


class ImageTarget(luigi.LocalTarget):
    def write(self, img):
        img.save(self.fn)

    def read(self):
        return Image.open(self.fn)

    def open(self):
        return self.read()

    def exists(self):
        return Path(self.path).exists()
