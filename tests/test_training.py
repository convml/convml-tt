import pytorch_lightning as pl

from convml_tt.system import (
    TripletTrainerModel,
    TripletTrainerDataModule,
    HeadFineTuner,
)
from convml_tt.data.examples import (
    fetch_example_dataset,
    ExampleData,
    PretrainedModel,
    fetch_pretrained_model,
)
from convml_tt.data.dataset import TileType, ImageSingletDataset
from convml_tt.data.transforms import get_transforms
from convml_tt.utils import get_embeddings
from convml_tt.external import fastai1_weights_loader


def test_train_new():
    trainer = pl.Trainer(max_epochs=5)
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=False, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch
    )
    trainer.fit(model=model, datamodule=datamodule)


def test_train_new_with_preloading():
    trainer = pl.Trainer(max_epochs=5)
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=False, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch, preload_data=True
    )
    trainer.fit(model=model, datamodule=datamodule)


def test_finetune_pretrained():
    trainer = pl.Trainer(max_epochs=5, callbacks=[HeadFineTuner()])
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=True, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=2, normalize_for_arch=arch
    )
    trainer.fit(model=model, datamodule=datamodule)


def test_load_from_weights():
    model_path = fetch_pretrained_model(
        pretrained_model=PretrainedModel.FIXED_NORM_STAGE2
    )
    model = fastai1_weights_loader.model_from_saved_weights(path=model_path)

    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    dataset = ImageSingletDataset(
        data_dir=data_path,
        stage="train",
        tile_type=TileType.ANCHOR,
        transform=get_transforms(step="predict", normalize_for_arch=model.base_arch),
    )
    get_embeddings(tile_dataset=dataset, model=model, prediction_batch_size=16)
