import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import random_split, DataLoader

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
        n_input_channels=4,
    ):
        super().__init__()
        ModelClass = getattr(tv_models, base_arch, None)

        if ModelClass is None:
            raise NotImplementedError(
                f"The `{base_arch}` model isn't available in pytorch-vision"
            )

        if l2_regularisation is not None:
            raise NotImplementedError()

        self.encoder = ModelClass(pretrained=pretrained)
        # We need to ensure the number of input channels matches the number of
        # channels in our data. If it doesn't we replace that layer here
        # (loosing the pretrained weights!)
        # NOTE: at least for the resnet models the convention is that the first
        # (the input) layer is `.conv1`.
        if self.encoder.conv1.in_channels != n_input_channels:
            conv1_orig = self.encoder.conv1
            self.encoder.conv1 = torch.nn.Conv2d(
                n_input_channels,
                conv1_orig.out_channels,
                kernel_size=conv1_orig.kernel_size,
                stride=conv1_orig.stride,
                padding=conv1_orig.padding,
                bias=conv1_orig.bias,
            )

        self.lr = lr
        self.margin = margin
        self.l2_regularisation = l2_regularisation

    def _loss(self, batch):
        y_anchor, y_neighbour, y_distant = [self.encoder(x) for x in batch]

        l_near = F.mse_loss(y_anchor, y_neighbour)
        l_distant = F.mse_loss(y_anchor, y_distant)
        loss = F.relu(l_near + l_distant + self.margin)
        return loss

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class TripletTrainerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_test_fraction=0.9, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        MNIST(self.data_dir, train=True, download=True)
        self.transform = None

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
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
