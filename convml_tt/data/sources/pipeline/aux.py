from pathlib import Path

import luigi
import numpy as np

from ....pipeline import YAMLTarget
from .. import DataSource
from ..goes16.pipeline import GOES16Query
from .scene_sources import GenerateSceneIDs, get_time_for_filename, parse_scene_id


class CheckForAuxiliaryFiles(luigi.Task):
    """
    Convenience task for downloading extra data for each scene. For now only
    GOES-16 fetching is implemented, not sure if others will be added in the
    future
    """

    data_path = luigi.Parameter(default=".")
    product_name = luigi.Parameter()

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        tasks = {}
        tasks["scene_ids"] = GenerateSceneIDs(data_path=self.data_path)

        data_source = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / data_source.source

        tasks["product"] = []
        if data_source.source == "goes16":
            aux_products = data_source._meta.get("aux_products", [])

            if self.product_name not in aux_products:
                raise Exception(
                    f"To fetch the `{self.product_name}` please add it to the meta info"
                )

            product = self.product_name

            for t_start, t_end in data_source.time_intervals:
                dt_total = t_end - t_start
                t_center = t_start + dt_total / 2.0

                t = GOES16Query(
                    data_path=source_data_path,
                    time=t_center,
                    dt_max=dt_total / 2.0,
                    channel=None,
                    product=product,
                )
                tasks["product"].append(t)

        elif data_source.source == "LES":
            pass
        else:
            raise NotImplementedError(data_source.source)

        return tasks

    def run(self):
        inputs = self.input()
        scene_ids = list(inputs.pop("scene_ids").open().keys())
        scene_times = np.array([parse_scene_id(scene_id)[1] for scene_id in scene_ids])

        data_source = self.data_source

        product = self.product_name
        product_inputs = inputs["product"]

        for product, product_inputs in inputs.items():
            product_times = []
            product_filenames = []
            for product_input in product_inputs:
                for fn_product in product_input.open():
                    t_file = get_time_for_filename(
                        data_source=data_source, filename=fn_product
                    )
                    product_times.append(t_file)
                    product_filenames.append(fn_product)

            product_times = np.array(product_times)

            product_fn_for_scenes = {}
            for scene_id, scene_time in zip(scene_ids, scene_times):
                i = np.argmin(np.abs(scene_time - product_times))
                product_fn = product_filenames[i]
                product_fn_for_scenes[scene_id] = product_fn

            self.output().write(product_fn_for_scenes)

    def output(self):
        data_source = self.data_source
        output = None

        if data_source.source == "goes16":
            fn = f"{self.product_name}.yml"
            p = Path(self.data_path) / "source_data" / data_source.source / "aux" / fn
            output = YAMLTarget(str(p))

        return output


class CheckForAllAuxiliaryFiles(luigi.WrapperTask):
    """
    Convenience task for downloading extra data for each scene. For now only
    GOES-16 fetching is implemented, not sure if others will be added in the
    future
    """

    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        tasks = {}

        data_source = self.data_source

        if data_source.source == "goes16":
            aux_products = data_source._meta.get("aux_products", [])

            for product in aux_products:
                tasks[product] = CheckForAuxiliaryFiles(
                    data_path=self.data_path, product_name=product
                )

        elif data_source.source == "LES":
            pass
        else:
            raise NotImplementedError(data_source.source)

        return tasks
