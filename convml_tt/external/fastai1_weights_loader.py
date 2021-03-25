import warnings

import torch
from torch import nn
from .nn_layers import AdaptiveConcatPool2d

from ..system import TripletTrainerModel


def model_from_saved_weights(path):
    """
    This routine is only here so that models which were trained with
    `convml_tt` based on fastai v1 can be loaded. In general for saving/loading
    pytorch-lightning's `trainer.save_checkpoint(...)` and
    `TripletTrainerModel.load_checkpoint(...)` should be used.
    """
    # pytorch may issue some warnings here, but we just want to load the
    # model anyway, so hide the wwarnings for now
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded_encoder = torch.load(path)
    batch_size = 5
    nx = ny = 256
    # the first layer is the conv1d, find out how many input channels it has
    n_input_channels = list(loaded_encoder)[0][0].in_channels

    # before we can run the model we need to add the adaptive-pooling layer and
    # the flattening layer in again because these couldn't be pickled (they
    # weren't part of pytorch 1.0.1 that the old model was created in)
    head = loaded_encoder[-1]
    new_head = nn.Sequential(AdaptiveConcatPool2d(size=1), nn.Flatten(), *head)
    loaded_encoder = nn.Sequential(*loaded_encoder[:-1], new_head)

    rand_batch = torch.rand((batch_size, n_input_channels, nx, ny))
    try:
        # check that the model accepts data shaped like a batch and produces the expected output
        result_batch = loaded_encoder(rand_batch)
        n_embedding_dims = result_batch.shape[-1]
        expected_shape = (batch_size, n_embedding_dims)
        if result_batch.shape != expected_shape:
            raise Exception(
                "The shape of the output of the loaded encoder doesn't have "
                f"the expected shape, {result_batch.shape} != {expected_shape}"
            )
        # all we know is the weights, so this model won't be possible to train further
        model = TripletTrainerModel(
            base_arch="resnet18",
            margin="unknown",
            lr="unknown",
            l2_regularisation="unknown",
        )
        model.encoder = loaded_encoder
        print(f"Weights loaded from `{path}`")
        return model
    except Exception as e:  # noqa
        print("There was a problem with loading the model weights:")
        raise
