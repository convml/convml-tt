# coding: utf-8
"""
Utility to strip pytorch-only model from fastai v1 `Learner` so the weights can
be unpickled and we can create a pytorch model from the same weights without
using fastai
"""

from torch import nn
import torch
from fastai.basic_train import load_learner
from convml_tt.external.fastai import AdaptiveConcatPool2d, Flatten


def get_pytorch_model(model):
    # the torch model is contained inside the fastai one as `model`
    tmodel = model.model

    # the head contains fastai-specific layers tha we can't unpickle unless
    # fastai is installed
    head = tmodel[-1]

    new_head_layers = []
    for n in range(len(head)):
        layer = head[n]
        if type(layer).__module__.startswith("fastai"):
            print(type(layer).__name__)
            if type(layer).__name__ == "AdaptiveConcatPool2d":
                new_layer = AdaptiveConcatPool2d(layer.ap.output_size)
            elif type(layer).__name__ == "Flatten":
                new_layer = Flatten()
            else:
                raise NotImplementedError(type(layer))

        new_head_layers.append(new_layer)
    new_head = nn.Sequential(*new_head_layers)

    return nn.Sequential(*tmodel[:-1], new_head)


def main(fastai_model_filename):
    torch_model_filename = fastai_model_filename.replace(".pkl", ".torch.pkl")
    model = load_learner(".", fastai_model_filename)
    tmodel = get_pytorch_model(model)
    torch.save(tmodel, torch_model_filename)
    print(f"pure-torch model saved to `{torch_model_filename}`")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("fastai_model_filename")
    args = argparser.parse_args()

    main(args.fastai_model_filename)
