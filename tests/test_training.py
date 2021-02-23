import pytorch_lightning as pl

from convml_tt.system import Tile2Vec, TripletTrainerDataModule, get_single_tile_dataset
from convml_tt.data.examples import (
    fetch_example_dataset,
    ExampleData,
    PretrainedModel,
    fetch_pretrained_model,
)
from convml_tt.data.dataset import TileType
from convml_tt.utils import get_embeddings


def test_train_new():
    trainer = pl.Trainer(max_epochs=5)
    model = Tile2Vec(pretrained=True)
    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(data_dir=data_path, batch_size=2)
    trainer.fit(model=model, datamodule=datamodule)


def test_load_from_weights():
    model_path = fetch_pretrained_model(
        pretrained_model=PretrainedModel.FIXED_NORM_STAGE2
    )
    model = Tile2Vec.from_saved_weights(path=model_path)

    data_path = fetch_example_dataset(dataset=ExampleData.TINY10)
    dataset = get_single_tile_dataset(
        data_dir=data_path, stage="train", tile_type=TileType.ANCHOR
    )
    get_embeddings(tile_dataset=dataset, model=model, prediction_batch_size=16)
