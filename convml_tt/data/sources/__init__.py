import pprint
from pathlib import Path
import yaml
import dateutil.parser
import datetime


def load_meta(dataset_path):
    path_abs = Path(dataset_path).expanduser().absolute()
    p = path_abs / "meta.yaml"
    name = p.parent.name
    with open(str(p)) as fh:
        meta = yaml.load(fh.read())
    meta["name"] = name
    meta["data_path"] = path_abs
    return meta


def _parse_datetime(o):
    if not type(o) == datetime.datetime:
        return dateutil.parser.parse(o)
    else:
        return o


class DataSource:
    def __init__(self, *args, **kwargs):
        self._meta = kwargs

        time_meta = self._meta["time"]
        self.t_start = _parse_datetime(time_meta["t_start"])
        if "N_days" in time_meta:
            duration = datetime.timedelta(days=time_meta["N_days"])
            self.t_end = self.t_start + duration
        elif "t_end" in time_meta:
            self.t_end = _parse_datetime(time_meta["t_end"])
        else:
            raise NotImplementedError(time_meta["time"])

        assert "source" in self._meta
        assert "type" in self._meta

    @classmethod
    def load(cls, path):
        path_abs = Path(path).expanduser().absolute()
        p = path_abs / "meta.yaml"
        name = p.parent.name
        with open(str(p)) as fh:
            meta = yaml.load(fh.read())
        meta["name"] = name
        meta["data_path"] = path_abs
        return cls(**meta)

    @property
    def source(self):
        return self._meta["source"]

    @property
    def type(self):
        return self._meta["type"]

    def __repr__(self):
        return pprint.pformat(
            {k: v for k, v in self._meta.items() if not k.startswith("_")}
        )

    def filter_scenes_by_time(self, scene_times):
        """
        Apply the time filtering specified for this source dataset if one is specified
        """
        if "N_hours_from_zenith" in self._meta["time"]:
            # TODO: implement filtering here
            pass
        return scene_times
