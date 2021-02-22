"""
Example datasets for use with the triplet-trainer
"""
import enum
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive

_URL_ROOT = "http://homepages.see.leeds.ac.uk/~earlcd/ml-datasets"


class ExampleData(enum.Enum):
    TINY10 = "Nx256_s200000.0_N0study_N10train"
    SMALL100 = "Nx256_s200000.0_N0study_N100train"
    LARGE2000S500 = "Nx256_s200000.0_N500study_N2000train"


# datasets tar-balls with their md5 hash
_checks = {}
_checks[ExampleData.TINY10] = "d094cd1b25408517259fc8d8dad63f05"
_checks[ExampleData.SMALL100] = "f45f9da7aa77b82e493c3289ea1ea951"
_checks[ExampleData.LARGE2000S500] = "bdc6184db155c99411c2d401794a41ec"


def get_example_dataset(dataset: ExampleData, data_dir="data/"):
    """
    Downloads example data and returns the path to it
    """
    url = f"{_URL_ROOT}/{dataset.value}.tgz"
    download_and_extract_archive(
        url=url,
        download_root=data_dir,
        md5=_checks[dataset],
    )

    return Path(data_dir) / dataset.value
