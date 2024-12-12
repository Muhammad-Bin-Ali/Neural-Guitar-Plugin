import torch
import torch.nn as nn
import pytorch_lightning as lightning
from torch.utils.data import TensorDataset, DataLoader

import pickle

import WaveNetModified

from dataclasses import dataclass


@dataclass
class HyperParams:
    num_channels: int
    dilation: int
    num_repeat: int
    kernel_size: int
    learning_rate: int
    data: str
    batch_size: int


class TrainingUtils:
    @staticmethod
    def pre_emphasis_filter(input, coeff=0.95):
        return torch.cat(input[:, :, 0:1], input[:, :, 1:] - coeff * input[:, :, :-1])

    @staticmethod
    def error_to_signal_ratio(predicted, actual):
        y = TrainingUtils.pre_emphasis_filter(actual)
        y_hat = TrainingUtils.pre_emphasis_filter(predicted)

        return (y - y_hat).pow(2).sum(dim=2) / y.pow(2).sum(dim=2)


class GuitarNeuraleNet(lightning.LightningModule):
    def __init__(self, hyper_params: HyperParams):
        super().__init__()
        self.wavenet = WaveNetModified(hyper_params.num_channels, hyper_params.dilation, hyper_params.num_repeat, hyper_params.kernel_size)
        self.hyper_params = hyper_params

    def forward(self, input):
        return self.wavenet(input)

    def configure_optimizers(self):
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.hyper_params.learning_rate)

    def training_step(self, batch, batch_idx):
        x, actual = batch
        predicted = self.forward(x)
        loss = TrainingUtils.error_to_signal_ratio(predicted, actual[:, :, -predicted.size(2) :]).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, actual = batch
        predicted = self.forward(x)
        loss = TrainingUtils.error_to_signal_ratio(predicted, actual[:, :, -predicted.size(2) :]).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = pickle.load(open(self.hyper_params.data, "rb"))
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hyper_params.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size, num_workers=4)
