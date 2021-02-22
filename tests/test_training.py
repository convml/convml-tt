import pytorch_lightning as pl

from convml_tt.system import Tile2Vec, TripletTrainerDataModule
from convml_tt.data.examples import get_example_dataset, ExampleData


def test_load_data_and_train():
    trainer = pl.Trainer(max_epochs=5)
    model = Tile2Vec(pretrained=True)
    data_path = get_example_dataset(dataset=ExampleData.TINY10)
    datamodule = TripletTrainerDataModule(data_dir=data_path, batch_size=2)
    trainer.fit(model=model, datamodule=datamodule)
