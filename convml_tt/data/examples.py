"""
Example datasets for use with the triplet-trainer
"""
import enum
from pathlib import Path
from typing import Union

from torchvision.datasets.utils import download_and_extract_archive

_URL_ROOT = "http://homepages.see.leeds.ac.uk/~earlcd/ml-datasets"


class ExampleData(enum.Enum):
    TINY10 = "Nx256_s200000.0_N0study_N10train"
    SMALL100 = "Nx256_s200000.0_N0study_N100train"
    LARGE2000S500 = "Nx256_s200000.0_N500study_N2000train"


class PretrainedModel(enum.Enum):
    FIXED_NORM_STAGE2 = "fixednorm-stage-2"


# datasets tar-balls with their md5 hash
_checks = {}
_checks[ExampleData.TINY10] = "d094cd1b25408517259fc8d8dad63f05"
_checks[ExampleData.SMALL100] = "75b45c9f368c298685dd88018eeb4f80"
_checks[ExampleData.LARGE2000S500] = "7a128c930d97059f0796b736164a721f"
_checks[PretrainedModel.FIXED_NORM_STAGE2] = "67c9db80576128128d0a355364abd311"


def _fetch_example(item: Union[ExampleData, PretrainedModel], data_dir="data/"):
    """
    Downloads example data and returns the path to it
    """
    url = f"{_URL_ROOT}/{item.value}.tgz"
    download_and_extract_archive(
        url=url,
        download_root=data_dir,
        md5=_checks[item],
    )
    return Path(data_dir)


def fetch_example_dataset(dataset: ExampleData, data_dir="data/"):
    """
    Downloads example data and returns the path to it
    """
    return _fetch_example(item=dataset, data_dir=data_dir) / dataset.value


def fetch_pretrained_model(pretrained_model: PretrainedModel, data_dir="data/"):
    """
    Downloads pretrained model and returns the path to it
    """
    fname = f"{pretrained_model.value}.torch.pkl"
    return _fetch_example(item=pretrained_model, data_dir=data_dir) / fname
