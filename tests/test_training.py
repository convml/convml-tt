import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from convml_tt.data.dataset import ImageSingletDataset, TileType
from convml_tt.data.examples import (
    ExampleData,
    PretrainedModel,
    fetch_example_dataset,
    load_pretrained_model,
)
from convml_tt.data.transforms import get_transforms
from convml_tt.system import (
    HeadFineTuner,
    TripletTrainerDataModule,
    TripletTrainerModel,
)
from convml_tt.trainer_onecycle import OneCycleTrainer
from convml_tt.utils import get_embeddings

if torch.cuda.is_available():
    DEVICES = 1
else:
    DEVICES = None


def test_train_new():
    trainer = pl.Trainer(max_epochs=5, devices=DEVICES)
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=False, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch
    )
    trainer.fit(model=model, datamodule=datamodule)


def test_train_new_anti_aliased():
    trainer = pl.Trainer(max_epochs=5, devices=DEVICES)
    arch = "resnet18"
    model = TripletTrainerModel(
        pretrained=False, base_arch=arch, anti_aliased_backbone=True
    )
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch
    )
    trainer.fit(model=model, datamodule=datamodule)


def test_train_new_with_preloading():
    trainer = pl.Trainer(max_epochs=5, devices=DEVICES)
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=False, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch, preload_data=True
    )
    trainer.fit(model=model, datamodule=datamodule)


def test_finetune_pretrained():
    trainer = pl.Trainer(max_epochs=5, callbacks=[HeadFineTuner()], devices=DEVICES)
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=True, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch
    )
    trainer.fit(model=model, datamodule=datamodule)


def assert_models_equal(model_1, model_2):
    differences = {}
    for (k1, v1), (k2, v2) in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(v1, v2):
            pass
        else:
            if k1 == k2:
                differences[k1] = (v1, v2)
            else:
                raise Exception

    if len(differences) > 0:
        msg = (
            f"There were differences found in {len(differences)} out of "
            f"{len(model_1.state_dict())} layers: " + ", ".join(differences.keys())
        )
        raise Exception(msg)


def test_load_from_weights():
    model = load_pretrained_model(pretrained_model=PretrainedModel.FIXED_NORM_STAGE2)

    # there was a bug where fetching, loading and producing embeddings the same
    # way again yielded different embeddings. I need to check that using a
    # loaded network always gives the same result
    model2 = load_pretrained_model(pretrained_model=PretrainedModel.FIXED_NORM_STAGE2)

    assert_models_equal(model, model2)

    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    dataset = ImageSingletDataset(
        data_dir=data_path,
        stage="train",
        tile_type=TileType.ANCHOR,
        transform=get_transforms(step="predict", normalize_for_arch=model.base_arch),
    )
    da_emb = get_embeddings(tile_dataset=dataset, model=model, prediction_batch_size=16)

    da_emb2 = get_embeddings(
        tile_dataset=dataset, model=model2, prediction_batch_size=16
    )

    np.testing.assert_allclose(da_emb, da_emb2)


def test_train_new_onecycle():
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = OneCycleTrainer(max_epochs=5, callbacks=[lr_monitor], devices=DEVICES)
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=False, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch
    )
    trainer.fit(model=model, datamodule=datamodule)
