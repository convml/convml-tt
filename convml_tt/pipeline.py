import coloredlogs
import luigi
import xarray as xr
import yaml

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


class YAMLTarget(luigi.LocalTarget):
    def write(self, obj):
        with self.open("w") as fh:
            yaml.dump(obj, fh)

    def read(self):
        with self.open() as fh:
            return yaml.load(fh)
