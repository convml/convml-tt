from pathlib import Path
from ...pipeline import XArrayTarget, DBTarget
from . import DataSource

import xarray as xr
import luigi
import datetime


class LESDataFile(luigi.Task):
    file_path = luigi.Parameter()

    def output(self):
        return XArrayTarget(self.file_path)


def _dt64_to_datetime(dt64):
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        return datetime.datetime.utcfromtimestamp(dt64.astype("O") / 1e9)


def find_les_files(data_path, filename_glob):
    les_data_path = Path(data_path) / "source_files"
    file_paths = list(les_data_path.glob(filename_glob))

    if len(file_paths) == 0:
        raise FileNotFoundError(
            f"No source datafiles found matching {les_data_path}/{filename_glob}"
        )
    return file_paths


class FindLESFiles(luigi.Task):
    """
    Check all the netCDF source files found for the `source_variable` and create
    yaml-file with all the filenames
    """

    data_path = luigi.Parameter()
    filename_glob = luigi.Parameter(default="*.nc")
    source_variable = luigi.Parameter()

    @staticmethod
    def get_time(filename):
        dt64 = xr.open_dataset(filename).time
        return _dt64_to_datetime(dt64)

    def requires(self):
        file_paths = find_les_files(
            data_path=self.data_path, filename_glob=self.filename_glob
        )

        tasks = [
            LESDataFile(file_path=str(file_path.absolute())) for file_path in file_paths
        ]
        return tasks

    @property
    def data_source(self):
        # TODO: quick hack so that we can filter by time before we split into
        # separate files
        return DataSource.load(path=Path(self.data_path) / ".." / "..")

    def run(self):
        filenames = []
        times = []
        for inp in self.input():
            ds_or_da = inp.open()
            file_path = inp.fn
            if isinstance(ds_or_da, xr.Dataset):
                ds = ds_or_da
                if self.source_variable not in ds.data_vars:
                    raise Exception(
                        f"Requested variable `{self.source_variable}` not found in"
                        f" datafile {file_path}"
                    )
                da = ds[self.source_variable]
            else:
                da = ds_or_da
                if not da.name == self.source_variable:
                    raise Exception(
                        f"Requested variable `{self.source_variable}` not found in"
                        f" datafile {file_path}"
                    )

            data_source = self.data_source
            times = da.time.values
            timestep_counter = 0
            if len(times) > 1:
                # split into individual files
                fn_root = Path(file_path).name.replace(".nc", "")
                p_file_root = Path(file_path).parent
                for time in times:
                    if not data_source.valid_scene_time(time):
                        continue
                    dt = _dt64_to_datetime(dt64=time)
                    t_str = dt.isoformat().replace(":", "")
                    filename = f"{fn_root}_{t_str}.nc"
                    da_timestep = da.sel(time=time)
                    p_new = p_file_root / "scenes" / filename
                    p_new.parent.mkdir(exist_ok=True, parents=True)
                    da_timestep.to_netcdf(p_new)
                    filenames.append(str(p_new))
                    timestep_counter += 1
            else:
                filenames.append(filename)

        if len(filenames) == 0:
            raise Exception("No scenes found")

        self.output().write(filenames)

    def output(self):
        return DBTarget(
            path=str(self.data_path), db_name="files", db_type=self.data_source.db_type
        )
