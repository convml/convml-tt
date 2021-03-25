"""
Example on how to train convml_tt with logging on weights & biases
(https://wandb.ai)
"""
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import os

from convml_tt.system import TripletTrainerModel, TripletTrainerDataModule, HeadFineTuner
import convml_tt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", default=5, type=int)
    parser.add_argument(
        "--log-to-wandb",
        default=False,
        action="store_true",
        help="Log training to Weights & Biases",
    )
    parser.add_argument(
        "--gpus", type=int, help="Number of GPUs to use for training", default=0
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Name of the project this run should belong to in the logger",
        default="convml_tt",
    )

    if os.environ.get("FROM_CHECKPOINT"):
        parser.add_argument(
            "--model-from-checkpoint",
            type=Path,
            help="location of model checkpoint to start from",
            default=None,
        )
        temp_args, _ = parser.parse_known_args()
        model = TripletTrainerModel.load_from_checkpoint(temp_args.model_from_checkpoint)
    else:
        parser = TripletTrainerModel.add_model_specific_args(parser)

    parser = TripletTrainerDataModule.add_data_specific_args(parser)
    args = parser.parse_args()

    trainer_kws = dict()
    if args.log_to_wandb:
        trainer_kws["logger"] = WandbLogger(project=args.project)

    if "pretrained" in args and args.pretrained:
        trainer_kws["callbacks"] = [HeadFineTuner()]

    if args.gpus not in [0, 1]:
        # default to Distributed Data Parallel when training on multiple GPUs
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-data-parallel
        trainer_kws["accelerator"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(args, **trainer_kws)
    # pl.Lightningmodule doesn't have a `from_argparse_args` yet, so we call it
    # here ourselves
    if model is None:
        model = pl.utilities.argparse.from_argparse_args(TripletTrainerModel, args)

    datamodule = TripletTrainerDataModule.from_argparse_args(
        args, normalize_for_arch=model.base_arch
    )

    # make sure we log all the arguments to w&b, pytorch-lightning only saves
    # the model hyperparameters by default
    if "logger" in trainer_kws:
        trainer_kws["logger"].experiment.config.update(args)
        trainer_kws["logger"].experiment.config.update(
            {"convml_tt__version": convml_tt.__version__}
        )

    trainer.fit(model=model, datamodule=datamodule)
