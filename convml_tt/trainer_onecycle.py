import inspect

import pytorch_lightning as pl
import semver
import torch


class AddOneCycleSchedulerCallback(pl.Callback):
    def setup(self, trainer, pl_module, stage):
        max_lr = getattr(pl_module, "lr", None)
        if max_lr is None:
            raise Exception(
                "model is missing a learning-rate attribute which is used to"
                " set the cycle's maximum learning-rate"
            )

        # TODO: this doesn't handle the case where the data is associated with
        # `pl_module`
        datamodule = trainer.datamodule
        datamodule.setup(
            stage="fit"
        )  # need to do this to ensure data-loader has been initiated
        n_steps_per_epoch = len(trainer.datamodule.train_dataloader())

        # TODO: take into account `max_steps` value for the scheduler
        should_be_none = ["min_steps", "max_steps"]
        for attr in should_be_none:
            value = getattr(trainer, attr)
            target_value = None
            if attr == "max_steps" and semver.compare(pl.__version__, "1.5.0") >= -1:
                # from v1.5.0 the default value for `max_steps` in
                # pytorch-lightning was changed to -1
                # https://github.com/PyTorchLightning/pytorch-lightning/pull/9460
                target_value = -1

            if value != target_value:
                raise Exception(
                    f"{attr} (== {value}) should be == `{target_value}` on the "
                    "trainer to use one-cycle training"
                )

        n_epochs = trainer.max_epochs

        optimizer = pl_module.configure_optimizers()

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise Exception(
                "To use one-cycle training `model.configure_optimizers()` should"
                " only return a single optimizer and no scheduler"
            )

        # build a one-cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            epochs=n_epochs,
            steps_per_epoch=n_steps_per_epoch,
        )

        # this will be the new `configure_optimizers` function on `pl_module`
        def configure_optimizers():
            # the scheduler expects to be called on each training step, so we
            # need to ensure that
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(scheduler=scheduler, interval="step"),
            )

        setattr(pl_module, "configure_optimizers", configure_optimizers)


class OneCycleTrainer(pl.Trainer):
    """
    Trainer which replaces the optimizer returned by the model's `get_optimizers()`
    with a one-cycle learning-rate scheduler together with the same optimizer.

    The model is assumed to only return an optimizer from `get_optimizers()`
    and the `max_epochs` values will be taken as the number of epochs in the
    cycle
    """

    def __init__(self, *args, **kwargs):
        callbacks = kwargs.pop("callbacks", [])
        callbacks = callbacks + [AddOneCycleSchedulerCallback()]
        super().__init__(callbacks=callbacks, *args, **kwargs)

    # need to copy the signature, not just because it's nicer for the user, but
    # because the `from_argparse_args` uses the signature to decide which
    # arguments to pass in
    __init__.__signature__ = inspect.signature(pl.Trainer.__init__)
