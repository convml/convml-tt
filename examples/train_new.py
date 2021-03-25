"""
Example on how to train convml_tt with logging on weights & biases
(https://wandb.ai)
"""
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from convml_tt.system import TripletTrainerModel, TripletTrainerDataModule
from convml_tt.data.examples import (
    fetch_example_dataset,
    ExampleData,
)


def main():
    logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=5, logger=logger)
    arch = "resnet18"
    model = TripletTrainerModel(pretrained=False, base_arch=arch)
    data_path = fetch_example_dataset(dataset=ExampleData.SMALL100)
    datamodule = TripletTrainerDataModule(
        data_dir=data_path, batch_size=8, normalize_for_arch=arch
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
