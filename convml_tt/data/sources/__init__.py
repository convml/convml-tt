import pprint
from pathlib import Path
import yaml
import dateutil.parser
import datetime

from .sampling.domain import LocalCartesianDomain, SourceDataDomain


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
            required_vars = ["tile_N", "tile_size", "N_triplets"]
            triplets_meta = sampling_meta["triplets"]
            if triplets_meta is None:
                triplets_meta = {}

            if not "scene_collections_splitting" in triplets_meta:
                triplets_meta[
                    "scene_collections_splitting"
                ] = "random_by_relative_sample_size"

            missing_vars = list(filter(lambda v: v not in triplets_meta, required_vars))
            if len(missing_vars) > 0:
                raise Exception(
                    f"To make triplet samplings you must also define the following variables "
                    f" {', '.join(missing_vars)}"
                )

            N_triplets = triplets_meta.get("N_triplets", {})
            assert "study" in N_triplets and "train" in N_triplets
            assert "tile_N" in triplets_meta
            assert "tile_size" in triplets_meta
            assert "scene_collections_splitting" in triplets_meta
            assert sum(N_triplets.values()) > 0

        self.sampling = sampling_meta

    def _parse_time_meta(self):
        time_meta = self._meta.get("time")
        if time_meta is None:
            return

        self.t_start = _parse_datetime(time_meta["t_start"])
        if "N_days" in time_meta:
            duration = datetime.timedelta(days=time_meta["N_days"])
            self.t_end = self.t_start + duration
        elif "t_end" in time_meta:
            self.t_end = _parse_datetime(time_meta["t_end"])
        else:
            raise NotImplementedError(time_meta["time"])

    @classmethod
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

    def filter_scenes_by_time(self, scene_times):
        """
        Apply the time filtering specified for this source dataset if one is specified
        """
        if "N_hours_from_zenith" in self._meta["time"]:
            # TODO: implement filtering here
            pass
        return scene_times
