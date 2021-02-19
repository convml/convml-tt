import pytorch_lightning as pl

from convml_tt.system import Tile2Vec, TripletTrainerDataModule

DATA_PATH = "/home/earlcd/git-repos/convml_tt/data/Nx256_s200000.0_N0study_N100train"


def test_load_data_and_train():
    trainer = pl.Trainer()
    model = Tile2Vec()
    datamodule = TripletTrainerDataModule(data_dir=DATA_PATH)
    trainer.fit(model=model, datamodule=datamodule)
