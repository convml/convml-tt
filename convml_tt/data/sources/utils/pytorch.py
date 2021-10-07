from pathlib import Path

from ... import dataset as pytorch_dataset
from ..pipeline import SceneRegriddedData, GenerateTiles


def get_pytorch_dataset(datasource, dataset_type, transform=None, **kwargs):
    """
    For this datasource return a pytorch.Dataset for a given dataset type
    coming from this datasource
    """
    if dataset_type == "triplets":
        datasource_path = datasource._meta["data_path"]
        dataset_task = GenerateTiles(data_path=datasource_path)
        fp_dataset = Path(dataset_task.output()).parent
        DatasetClass = pytorch_dataset.ImageTripletDataset
    elif dataset_type == "rect":
        if "scene_id" not in kwargs:
            raise Exception(
                "For the moment `scene_id` must be given when creating a pytorch.Dataset for rect domains"
            )
        scene_id = kwargs.pop("scene_id")
        datasource_path = datasource._meta["data_path"]
        dataset_task = SceneRegriddedData(data_path=datasource_path, scene_id=scene_id)
        fp_dataset = Path(dataset_task.output()).parent
        DatasetClass = pytorch_dataset.MovingWindowImageTilingDataset
    else:
        raise NotImplementedError(dataset_type)

    if not dataset_task.output().exists():
        raise Exception(
            f"The pipeline to produce data for `{dataset_task}` hasn't been run yet"
        )

    dataset = DatasetClass(data_dir=fp_dataset, transform=transform, **kwargs)
    return dataset
