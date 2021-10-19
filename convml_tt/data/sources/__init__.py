import pprint
from pathlib import Path
import yaml
import dateutil.parser
import datetime
import numpy as np
import functools

from .sampling.domain import LocalCartesianDomain, SourceDataDomain
from .utils import time_filters


def _parse_datetime(o):
    if not type(o) == datetime.datetime:
        return dateutil.parser.parse(o)
    else:
        return o


def _parse_time_intervals(time_meta):
    if "intervals" in time_meta:
        for time_interval_meta in time_meta["intervals"]:
            for time_interval in _parse_time_intervals(time_meta=time_interval_meta):
                yield time_interval
    else:
        t_start = _parse_datetime(time_meta["t_start"])
        if "N_days" in time_meta:
            duration = datetime.timedelta(days=time_meta["N_days"])
            t_end = t_start + duration
        elif "t_end" in time_meta:
            t_end = _parse_datetime(time_meta["t_end"])
        else:
            raise NotImplementedError(time_meta["time"])

        yield (t_start, t_end)


class DataSource:
    def __init__(self, *args, **kwargs):
        self._meta = kwargs

        self._parse_time_meta()
        self._parse_sampling_meta()
        self._parse_domain_meta()

        assert "source" in self._meta
        assert "type" in self._meta

    def _parse_domain_meta(self):
        domain_meta = self._meta.get("domain", dict(kind="as_source"))
        local_cart_reqd_fields = [
            "central_latitude",
            "central_longitude",
            "l_zonal",
            "l_meridional",
        ]

        if all([field in domain_meta for field in local_cart_reqd_fields]):
            kwargs = {field: domain_meta[field] for field in local_cart_reqd_fields}
            domain = LocalCartesianDomain(**kwargs)
        elif domain_meta.get("kind") == "as_source":
            domain = SourceDataDomain()
        else:
            raise NotImplementedError(domain_meta)

        self.domain = domain

    def _parse_sampling_meta(self):
        sampling_meta = self._meta.get("sampling", {})

        if "triplets" in sampling_meta:
            required_vars = ["tile_size", "N_triplets"]
            triplets_meta = sampling_meta["triplets"]
            if triplets_meta is None:
                triplets_meta = {}

            if "scene_collections_splitting" not in triplets_meta:
                triplets_meta[
                    "scene_collections_splitting"
                ] = "random_by_relative_sample_size"

            # default tile is 256x256 pixels
            if "tile_N" not in triplets_meta:
                triplets_meta["tile_N"] = 256

            missing_vars = list(filter(lambda v: v not in triplets_meta, required_vars))
            if len(missing_vars) > 0:
                raise Exception(
                    f"To make triplet samplings you must also define the following variables "
                    f" {', '.join(missing_vars)}"
                )

            N_triplets = triplets_meta.get("N_triplets", {})

            # the default triplets collection is called "train"
            if type(N_triplets) == int:
                triplets_meta["N_triplets"] = dict(train=N_triplets)

            assert "train" in triplets_meta["N_triplets"]
            assert "tile_N" in triplets_meta
            assert "tile_size" in triplets_meta
            assert "scene_collections_splitting" in triplets_meta
            assert sum(triplets_meta["N_triplets"].values()) > 0

        self.sampling = sampling_meta

    def _parse_time_meta(self):
        time_meta = self._meta.get("time")
        if time_meta is None:
            if self.source == "goes16":
                raise Exception(
                    "The goes16 data source requires that you define the start "
                    "time (t_start) and either end time (t_end) or number of days "
                    "(N_days) in a `time` section of `meta.yaml`"
                )
            return
        else:
            self._time_intervals = list(_parse_time_intervals(time_meta=time_meta))

    @property
    def time_intervals(self):
        return self._time_intervals

    @classmethod
    @functools.lru_cache(maxsize=10)
    def load(cls, path):
        path_abs = Path(path).expanduser().absolute()
        p = path_abs / "meta.yaml"
        name = p.parent.name
        with open(str(p)) as fh:
            meta = yaml.load(fh.read(), Loader=yaml.FullLoader)
        if meta is None:
            raise Exception(
                "Please as minimum define the `source` and `type` of this "
                "datasource in `meta.yaml`"
            )
        meta["name"] = name
        meta["data_path"] = path_abs
        return cls(**meta)

    @property
    def source(self):
        return self._meta["source"]

    @property
    def type(self):
        return self._meta["type"]

    @property
    def files(self):
        return self._meta.get("files", None)

    def __repr__(self):
        return pprint.pformat(
            {k: v for k, v in self._meta.items() if not k.startswith("_")}
        )

    def valid_scene_time(self, scene_time):
        """
        Apply the time filtering specified for this source dataset if one is specified
        """

        filters = self._meta["time"].get("filters", {})
        for filter_kind, filter_value in filters.items():
            if filter_kind == "N_hours_from_zenith":
                lon_zenith = self.domain.central_longitude
                filter_fn = functools.partial(
                    time_filters.within_dt_from_zenith,
                    dt_zenith_max=datetime.timedelta(hours=filter_value),
                    lon_zenith=lon_zenith,
                )
            elif filter_kind == time_filters.DATETIME_ATTRS:
                filter_fn = functools.partial(
                    time_filters.within_attr_values, **{filter_kind: filter_value}
                )
            else:
                raise NotImplementedError(filter_kind)

            if not filter_fn(scene_time):
                return False

        return True
