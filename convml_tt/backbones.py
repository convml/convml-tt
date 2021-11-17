"""
Copy of [lightning-flash](https://github.com/PyTorchLightning/lightning-flash)'s
`flash/vision/backbones.py` as of commit `24c5b66e`, duplicated here so we
don't have to install all of lightning-flash

As of commit 49de5a0b0f631bbdd5e136abc920c56a7ed14821 lightning-flash no longer
has a flash.vision.backbones module this was split into flash.image...

Also, pytorch-lightning doens't have a routine to check for `lightning-bolts`
so I've removed the bolts models.
"""
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple

import antialiased_cnns
import torchvision
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

ROOT_S3_BUCKET = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"

MOBILENET_MODELS = ["mobilenet_v2"]
VGG_MODELS = ["vgg11", "vgg13", "vgg16", "vgg19"]
RESNET_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
]
DENSENET_MODELS = ["densenet121", "densenet169", "densenet161"]
TORCHVISION_MODELS = MOBILENET_MODELS + VGG_MODELS + RESNET_MODELS + DENSENET_MODELS


def backbone_and_num_features(
    model_name: str,
    fpn: bool = False,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    anti_aliased=False,
    **kwargs,
) -> Tuple[nn.Module, int]:
    """
    Args:
        model_name: backbone supported by `torchvision`
        fpn: If True, creates a Feature Pyramind Network on top of Resnet based CNNs.
        pretrained: if true, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers: number of trainable resnet layers starting from final block.
        anti_alised: if True, loads an anti-aliased version of the the backbone (where supported)

    >>> backbone_and_num_features('mobilenet_v2')  # doctest: +ELLIPSIS
    (Sequential(...), 1280)
    >>> backbone_and_num_features('resnet50', fpn=True)  # doctest: +ELLIPSIS
    (BackboneWithFPN(...), 256)
    >>> backbone_and_num_features('swav-imagenet')  # doctest: +ELLIPSIS
    (Sequential(...), 2048)
    """
    if fpn:
        if anti_aliased:
            raise NotImplementedError(
                "anti-aliased versions of feature-pyramid models aren't currently available"
            )
        if model_name in RESNET_MODELS:
            backbone = resnet_fpn_backbone(
                model_name,
                pretrained=pretrained,
                trainable_layers=trainable_backbone_layers,
                **kwargs,
            )
            fpn_out_channels = 256
            return backbone, fpn_out_channels
        else:
            rank_zero_warn(
                f"{model_name} backbone is not supported with `fpn=True`, `fpn` won't be added."
            )

    if model_name in TORCHVISION_MODELS:
        return torchvision_backbone_and_num_features(
            model_name, pretrained, anti_aliased=anti_aliased
        )

    raise ValueError(f"{model_name} is not supported yet.")


def torchvision_backbone_and_num_features(
    model_name: str, pretrained: bool = True, anti_aliased=False
) -> Tuple[nn.Module, int]:
    """
    >>> torchvision_backbone_and_num_features('mobilenet_v2')  # doctest: +ELLIPSIS
    (Sequential(...), 1280)
    >>> torchvision_backbone_and_num_features('resnet18')  # doctest: +ELLIPSIS
    (Sequential(...), 512)
    >>> torchvision_backbone_and_num_features('densenet121')  # doctest: +ELLIPSIS
    (Sequential(...), 1024)
    """
    if anti_aliased:
        model = getattr(antialiased_cnns, model_name, None)
        if model is None:
            raise MisconfigurationException(
                f"an anti-alised version of {model_name} doesn't yet exist"
            )
    else:
        model = getattr(torchvision.models, model_name, None)
        if model is None:
            raise MisconfigurationException(
                f"{model_name} is not supported by torchvision"
            )

    if model_name in MOBILENET_MODELS + VGG_MODELS:
        model = model(pretrained=pretrained)
        backbone = model.features
        num_features = model.classifier[-1].in_features
        return backbone, num_features

    elif model_name in RESNET_MODELS:
        model = model(pretrained=pretrained)
        # remove the last two layers & turn it into a Sequential model
        backbone = nn.Sequential(*list(model.children())[:-2])
        num_features = model.fc.in_features
        return backbone, num_features

    elif model_name in DENSENET_MODELS:
        model = model(pretrained=pretrained)
        backbone = nn.Sequential(*model.features, nn.ReLU(inplace=True))
        num_features = model.classifier.in_features
        return backbone, num_features

    raise ValueError(f"{model_name} is not supported yet.")
