"""
Pipeline tasks for generating triplets from satellite data

Steps for GOES-16 data:

    1. Fetch all radiance files that fit within given time window and group by
       scene (same timestamp)
    2. Crop radiance files to domain and create RGB composite on native grid of
       observations
"""
from pathlib import Path
import random

import dateutil.parser
import luigi

from ....pipeline import YAMLTarget
from .goes.pipeline import GOES16Fetch
from ...dataset import ImageTripletDataset


TILE_FILENAME_FORMAT = ImageTripletDataset.TILE_FILENAME_FORMAT


def _get_datasource_task(datasource_name):
    if datasource_name == "goes16":
        return GOES16Fetch

    raise NotImplementedError(datasource_name)


class DatetimeListParameter(luigi.Parameter):
    def parse(self, x):
        return [dateutil.parser.parse(s) for s in x.split(",")]

    def serialize(self, x):
        return ",".join([t.isoformat() for t in x])


def pick_one_time_per_date_for_study(
    datasets_filenames, datasource_cli, ensure_each_day_has_training_data=False
):
    """
    Split the the datasets filenames so that the study set contains exactly one
    set of files for a dataset on each day
    """
    dataset_files_by_date = {}

    for fns in datasets_filenames:
        date = datasource_cli.parse_key(str(fns[0]), parse_times=True)[
            "start_time"
        ].date()
        dataset_files_by_date.setdefault(date, []).append(fns)

    def _split_date(datasets_filenames):
        datasets_filenames = list(datasets_filenames)
        if ensure_each_day_has_training_data and len(datasets_filenames) < 2:
            raise Exception(
                "There is only one dataset for the given date "
                "(`{}`), is this a mistake?".format(datasets_filenames[0][0])
            )
        random.shuffle(datasets_filenames)
        return datasets_filenames[:1], datasets_filenames[1:]

    datasets_study = []
    datasets_train = []
    for date in dataset_files_by_date:
        l_study_d, l_train_d = _split_date(dataset_files_by_date[date])
        datasets_study += l_study_d
        datasets_train += l_train_d

    return dict(train=datasets_train, study=datasets_study)


class StudyTrainSplit(luigi.Task):
    dt_max = luigi.FloatParameter()
    times = DatetimeListParameter()
    data_path = luigi.Parameter()
    datasource = luigi.Parameter()

    def requires(self):
        FetchTask = _get_datasource_task(datasource_name=self.datasource)
        return FetchTask(
            dt_max=self.dt_max,
            times=self.times,
            data_path=self.data_path,
        )

    def run(self):
        datasets_filenames_all = self.input().read()
        datasource_cli = _get_datasource_task(datasource_name=self.datasource).CLIClass
        datasets_filenames_split = pick_one_time_per_date_for_study(
            datasets_filenames_all,
            datasource_cli=datasource_cli,
        )
        Path(self.output().fn).parent.mkdir(exist_ok=True)
        self.output().write(datasets_filenames_split)

    def output(self):
        fn = "source_data/training_study_split.yaml"
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))


def generate_tile_triplets(
    scenes,
    tiling_bbox,
    tile_N,
    tile_size,
    output_dir,
    N_triplets,
    max_workers=4,
    neighbor_distant_frac=0.8,
    N_start=0,
):
    if len(scenes) < 2:
        raise Exception("Need at least two scenes")

    print("Generating tiles")

    for triplet_n in tqdm(range(N_triplets)[N_start:]):
        # sample different datasets
        tn_target, tn_dist = random.sample(range(len(scenes)), 2)
        da_target_scene = scenes[tn_target]
        da_distant_scene = scenes[tn_dist]

        prefixes = "anchor neighbor distant".split(" ")

        output_files_exist = [
            os.path.exists(output_dir / TRIPLET_FN_FORMAT.format(triplet_n, p))
            for p in prefixes
        ]

        if all(output_files_exist):
            continue

        tiles_and_imgs = tiler.triplet_generator(
            da_target_scene=da_target_scene,
            da_distant_scene=da_distant_scene,
            tile_size=tile_size,
            tile_N=tile_N,
            tiling_bbox=tiling_bbox,
            neigh_dist_scaling=neighbor_distant_frac,
        )

        tiles, imgs = zip(*tiles_and_imgs)

        for (img, prefix) in zip(imgs, prefixes):
            fn_out = TRIPLET_FN_FORMAT.format(triplet_n, prefix)
            img.save(output_dir / fn_out, "PNG")

        meta = dict(
            target=dict(
                source_files=da_target_scene.attrs["source_files"],
                anchor=tiles[0].serialize_props(),
                neighbor=tiles[1].serialize_props(),
            ),
            distant=dict(
                source_files=da_distant_scene.attrs["source_files"],
                loc=tiles[2].serialize_props(),
            ),
        )


def _generate_triplet_locs_for_scenes():
    scenes_bbox = {}
    # read in each scene's DataArray and extract its

    if da_distant_scene is None:
        while True:
            dist_loc = _perturb_loc(anchor_loc, scaling=distant_dist_scaling)
            if _point_valid(dist_loc):
                break
    else:

    locs = [anchor_loc, neighbor_loc, dist_loc]

    tiles = [Tile(lat0=lat, lon0=lon, size=tile_size) for (lon, lat) in locs]

    # create a list of the three scenes used for creating triplets
    da_scene_set = [da_target_scene, da_target_scene]
    if da_distant_scene is None:
        da_scene_set.append(da_target_scene)
    else:
        da_scene_set.append(da_distant_scene)

    # on each of the three scenes use the three tiles to create a resampled
    # image
    try:
        return [
            (tile, tile.create_true_color_img(da_scene, resampling_N=tile_N))
            for (tile, da_scene) in zip(tiles, da_scene_set)
        ]
    except Tile.TileBoundsOutsideOfInputException:
        return triplet_generator(
            da_target_scene,
            tile_size,
            tiling_bbox,
            tile_N,
            da_distant_scene,
            neigh_dist_scaling,
            distant_dist_scaling,
        )


def _write_tile_meta(tile_id, tile_type, data, append_existing=True):
    fn = TILE_FILENAME_FORMAT.format(tile_type=tile_type, tile_id=tile_id, ext="yaml")

    with open(str(output_dir/fn), "w") as fh:
        yaml.dump(data, fh, default_flow_style=False)



class GenerateTripletsMeta(luigi.Task):
    """
    Write meta-files for each tile to be produced
    """
    dt_max = luigi.FloatParameter()
    times = DatetimeListParameter()
    data_path = luigi.Parameter()
    datasource = luigi.Parameter()

    def requires(self):
        return StudyTrainSplit(
            dt_max=self.dt_max,
            times=self.times,
            data_path=self.data_path,
            datasource=self.datasource,
        )

    def run(self):
        pass

    def output(self):
        return YAMLTarget("training_study_split.yaml")
