from pathlib import Path
from ...pipeline import XArrayTarget, YAMLTarget

import xarray as xr
import luigi


class LESDataFile(luigi.Task):
    file_path = luigi.Parameter()

    def output(self):
        return XArrayTarget(self.file_path)


class FindLESFiles(luigi.Task):
    """
    Check all the netCDF source files found for the `source_variable` and create
    yaml-file with all the filenames
    """

    data_path = luigi.Parameter()
    filename_glob = luigi.Parameter(default="*.nc")
    source_variable = luigi.Parameter()

    def requires(self):
        data_path = Path(self.data_path)
        file_paths = list(data_path.glob(self.filename_glob))

        if len(file_paths) == 0:
            raise FileNotFoundError(
                f"No source datafiles found matching {self.data_path}/{self.filename_glob}"
            )

        tasks = [
            LESDataFile(file_path=str(file_path.absolute())) for file_path in file_paths
        ]
        return tasks

    def run(self):
        filenames = []
        for inp in self.input():
            ds_or_da = inp.open()
            filename = inp.fn
            if isinstance(ds_or_da, xr.Dataset):
                ds = ds_or_da
                if self.source_variable not in ds.data_vars:
                    raise Exception(
                        f"Requested variable `{self.source_variable}` not found in"
                        f" datafile {filename}"
                    )
            else:
                da = ds_or_da
                if not da.name == self.source_variable:
                    raise Exception(
                        f"Requested variable `{self.source_variable}` not found in"
                        f" datafile {filename}"
                    )
            filenames.append(filename)

        self.output().write(filenames)

    def output(self):
        fn = "files.yml"
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))
