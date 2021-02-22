import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import random_split, DataLoader
from flash.vision import backbones as flash_backbones

from .data.dataset import ImageTripletDataset

from torchvision.datasets import MNIST
from torchvision import transforms


class Tile2Vec(pl.LightningModule):
    def __init__(
        self,
        base_arch="resnet18",
        pretrained=False,
        margin=1.0,
        lr=1.0e-5,
        l2_regularisation=None,
        n_input_channels=3,
        n_embedding_dims=100,
    ):
        super().__init__()

        if l2_regularisation is not None:
            raise NotImplementedError()

        self.lr = lr
        self.margin = margin
        self.l2_regularisation = l2_regularisation
        self.n_embedding_dims = n_embedding_dims

        self.encoder = self._create_convolution_layers(
            n_input_channels=n_input_channels,
            base_arch=base_arch,
            pretrained=pretrained,
        )

    def _create_convolution_layers(
        self, n_input_channels, base_arch=None, pretrained=False
    ):
        pretrained = True
        encoder, n_features_encoder = flash_backbones.backbone_and_num_features(
            model_name=base_arch, pretrained=pretrained
        )

        # We need to ensure the number of input channels matches the number of
        # channels in our data. If it doesn't we replace that layer here
        # (loosing the pretrained weights!)
        # NOTE: at least for the resnet models the convention is that the first
        # (the input) layer is `.conv1`.
        layers = list(encoder.children())
        if not isinstance(layers[0], torch.nn.Conv2d):
            raise Exception(f"Recognised type for input layer {layers[0]}")

        input_conv_orig = layers[0]

        if input_conv_orig.in_channels != n_input_channels:
            layers[0] = torch.nn.Conv2d(
                n_input_channels,
                input_conv_orig.out_channels,
                kernel_size=input_conv_orig.kernel_size,
                stride=input_conv_orig.stride,
                padding=input_conv_orig.padding,
                bias=input_conv_orig.bias,
            )

            # make a new encoder with the right number of input layers
            encoder = torch.nn.Sequential(*layers)

        # make "head" block which takes features of the encoder (resnet18 has
        # 512), uses adaptive pooling to reduce the x- and y-dimension, and
        # then uses a fully-connected layer to make the desired output size
        head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),  # -> (batch_size, n_features, 1, 1)
            torch.nn.Flatten(),  # -> (batch_size, n_features)
            torch.nn.Linear(
                in_features=n_features_encoder, out_features=self.n_embedding_dims
            ),  # -> (batch_size, n_embedding_dims)
        )

        return torch.nn.Sequential(encoder, head)

    def _loss(self, batch):
        y_anchor, y_neighbour, y_distant = [self.encoder(x) for x in batch]

        # per-batch distance between anchor and neighbour tile in embedding
        # space, shape: (batch_size, )
        l_near = ((y_anchor - y_neighbour)**2.0).sum(dim=1)
        # per-batch distance between anchor and distant tile in embedding
        # space, shape: (batch_size, )
        l_distant = ((y_anchor - y_distant)**2.0).sum(dim=1)

        # total loss goes through a ReLU with the margin added, and we take
        # mean across the batch
        loss = torch.mean(F.relu(l_near - l_distant + self.margin))

        return loss

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class RemoveImageAlphaTransform:
    def __call__(self, x):
        return x[:3, :, :]


class TripletTrainerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_test_fraction=0.9, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        MNIST(self.data_dir, train=True, download=True)
        self.transform = None

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                RemoveImageAlphaTransform(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_test_fraction = train_test_fraction
        self.batch_size = batch_size
        self._train_dataset = None
        self._test_dataset = None

    def prepare_data(self, *args, **kwargs):
        # data already downloaded
        pass

    def setup(self, stage=None):
        if stage == "fit":
            full_dataset = ImageTripletDataset(
                data_dir=self.data_dir, kind="train", transform=self.transform
            )
            n_samples = len(full_dataset)
            n_train = int(n_samples * self.train_test_fraction)
            n_test = n_samples - n_train
            self._train_dataset, self._test_dataset = random_split(
                full_dataset, [n_train, n_test]
            )
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self):
        return DataLoader(dataset=self._train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(dataset=self._train_dataset, batch_size=self.batch_size)
