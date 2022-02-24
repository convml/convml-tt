"""
Example on how to train convml_tt with logging on weights & biases
(https://wandb.ai)
"""
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from . import __version__, trainer_logging
from .system import HeadFineTuner, TripletTrainerDataModule, TripletTrainerModel
from .trainer_onecycle import OneCycleTrainer


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs", default=5, type=int, help="number of epochs to train for"
    )
    parser.add_argument(
        "--log-to-wandb",
        default=False,
        action="store_true",
        help="Log training to Weights & Biases",
    )
    parser.add_argument(
        "--log-dendrogram",
        default=False,
        action="store_true",
        help="Log dendrogram plot before and after training",
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
    parser.add_argument(
        "--use-one-cycle-training",
        default=False,
        action="store_true",
        help="Use one-cycle learning rate scheduling",
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Floating point precision to use during training",
    )

    if os.environ.get("FROM_CHECKPOINT"):
        parser.add_argument(
            "--model-from-checkpoint",
            type=Path,
            help="location of model checkpoint to start from",
            default=None,
        )
        temp_args, _ = parser.parse_known_args()
        model = TripletTrainerModel.load_from_checkpoint(
            temp_args.model_from_checkpoint
        )
    else:
        parser = TripletTrainerModel.add_model_specific_args(parser)
        model = None

    parser = TripletTrainerDataModule.add_data_specific_args(parser)
    # show the default args when showing the usage
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    args = parser.parse_args(args=args)

    trainer_kws = dict()
    if args.log_to_wandb:
        trainer_kws["logger"] = WandbLogger(project=args.project, log_model=True)

    if "pretrained" in args and args.pretrained:
        trainer_kws["callbacks"] = [HeadFineTuner()]

    if args.gpus not in [0, 1]:
        # default to Distributed Data Parallel when training on multiple GPUs
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-data-parallel
        trainer_kws["accelerator"] = "ddp"

    if args.use_one_cycle_training:
        TrainerClass = OneCycleTrainer
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        trainer_kws["callbacks"] = [lr_monitor]
    else:
        TrainerClass = pl.Trainer

    trainer = TrainerClass.from_argparse_args(args, **trainer_kws)
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
            {"convml_tt__version": __version__}
        )

    dendrogram_logger = None
    if args.log_dendrogram:
        dendrogram_logger = trainer_logging.make_dendrogram_logger(
            datamodule=datamodule
        )
        dendrogram_logger(model=model, stage="start")

    trainer.fit(model=model, datamodule=datamodule)

    if dendrogram_logger:
        dendrogram_logger(model=model, stage="end")


if __name__ == "__main__":
    main()
