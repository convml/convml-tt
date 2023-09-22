import hashlib
import warnings

import torch
from torch import nn

from ..system import TripletTrainerModel
from .nn_layers import AdaptiveConcatPool2d


class ScalingLayer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def model_from_saved_weights(path):
    """
    This routine is only here so that models which were trained with
    `convml_tt` based on fastai v1 can be loaded. In general for saving/loading
    pytorch-lightning's `trainer.save_checkpoint(...)` and
    `TripletTrainerModel.load_checkpoint(...)` should be used.
    """
    dev = torch.device("cpu")

    # pytorch may issue some warnings here, but we just want to load the
    # model anyway, so hide the wwarnings for now
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded_encoder = torch.load(path).to(dev)
    batch_size = 5
    nx = ny = 256
    # the first layer is the conv1d, find out how many input channels it has
    n_input_channels = list(loaded_encoder)[0][0].in_channels

    # before we can run the model we need to add the adaptive-pooling layer and
    # the flattening layer in again because these couldn't be pickled (they
    # weren't part of pytorch 1.0.1 that the old model was created in). We also
    # introduce a scaling layer because the fastai v1 network created values
    # that we're very small
    model_hash = hashlib.md5(open(path, "rb").read()).hexdigest()
    if model_hash == "d23b8370173082774052974f4729733e":
        scaling = 1.0e3
    else:
        scaling = 1.0e0

    head = loaded_encoder[-1]
    new_head = nn.Sequential(
        AdaptiveConcatPool2d(size=1), nn.Flatten(), *head, ScalingLayer(scaling)
    )
    # we use the backbone as-is
    backbone = loaded_encoder[:-1]

    rand_batch = torch.rand((batch_size, n_input_channels, nx, ny)).to(dev)
    try:
        # check that the model accepts data shaped like a batch and produces the expected output
        # all we know is the weights, so this model won't be possible to train further
        model = TripletTrainerModel(
            base_arch="unknown",
            margin="unknown",
            lr="unknown",
            l2_regularisation="unknown",
        )
        # when passing in `base_arch="unknown"` above the backbone and head
        # won't be set on the model and we can set them directly here
        model.backbone = backbone
        model.head = new_head
        # we finally set the `base_arch` attribute manually as this is used for
        # working out the image normalization transform
        model.base_arch = "resnet18"
        # finally we set a flag which ensure the output is scaled by 1.0e3 (for some reason the model I trained
        setattr(model, "from_fastai_v1", True)

        result_batch = model.forward(rand_batch)
        n_embedding_dims = result_batch.shape[-1]
        expected_shape = (batch_size, n_embedding_dims)
        if result_batch.shape != expected_shape:
            raise Exception(
                "The shape of the output of the loaded encoder doesn't have "
                f"the expected shape, {result_batch.shape} != {expected_shape}"
            )
        print(f"Weights loaded from `{path}`")
        return model
    except Exception as e:  # noqa
        print("There was a problem with loading the model weights:")
        raise
