"""
Example datasources for use with the triplet-trainer
"""
import enum
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive

_URL_ROOT = "http://homepages.see.leeds.ac.uk/~earlcd/ml-datasources"


class ExampleDatasource(enum.Enum):
    EUREC4A_SMALL = "eurec4a-small"


# datasources tar-balls with their md5 hash
_checks = {}
_checks[ExampleDatasource.EUREC4A_SMALL] = "2e19801de21608968d107ac01ecf2a3b"


def _fetch_example(item: ExampleDatasource, data_dir="data/"):
    """
    Downloads example datasource and returns the path to it
    """
    url = f"{_URL_ROOT}/{item.value}.tgz"
    download_and_extract_archive(
        url=url,
        download_root=data_dir,
        md5=_checks[item],
    )
    return Path(data_dir)


def fetch_example_datasource(datasource: ExampleDatasource, data_dir="data/"):
    """
    Downloads example datasource and returns the path to it
    """
    return _fetch_example(item=datasource, data_dir=data_dir) / datasource.value


def main(args=None):
    """
    CLI interface for downloading data examples
    """
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("datasource_name", choices=[ds.name for ds in ExampleDatasource])
    argparser.add_argument("--path", default="data/", type=Path)
    args = argparser.parse_args(args=args)
    datasource_path = fetch_example_datasource(
        datasource=ExampleDatasource[args.datasource_name], data_dir=args.path
    )
    print(f"Downloaded example datasource `{args.datasource_name}` to `{datasource_path}`")


if __name__ == "__main__":
    main()
