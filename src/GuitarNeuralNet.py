import torch
import torch.nn as nn
import pytorch_lightning as lightning
from torch.utils.data import TensorDataset, DataLoader
import pickle
import WaveNetModified


class TrainingUtils:
    @staticmethod
    def pre_emphasis_filter(input, coeff=0.95):
        return torch.cat((input[:, :, 0:1], input[:, :, 1:] - coeff * input[:, :, :-1]), dim=2)

    @staticmethod
    def error_to_signal_ratio(predicted, actual):
        y = TrainingUtils.pre_emphasis_filter(actual)
        y_hat = TrainingUtils.pre_emphasis_filter(predicted)

        return (y - y_hat).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)  # so we're not dividing by 0


class GuitarNeuraleNet(lightning.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()
        self.wavenet = WaveNetModified.WaveNetModified(hyper_params["num_channels"], hyper_params["dilation_depth"], hyper_params["num_repeat"], hyper_params["kernel_size"])
        self.save_hyperparameters(hyper_params)
        self.validation_step_outputs = []

    def forward(self, input):
        return self.wavenet(input)

    def configure_optimizers(self):
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.hparams["learning_rate"])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            num_workers=4,
        )

    def training_step(self, batch, batch_idx):
        x, actual = batch
        predicted = self.forward(x)
        loss = TrainingUtils.error_to_signal_ratio(predicted, actual[:, :, -predicted.size(2) :]).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams["batch_size"], num_workers=4)

    def validation_step(self, batch, batch_idx):
        x, actual = batch
        predicted = self.forward(x)
        loss = TrainingUtils.error_to_signal_ratio(predicted, actual[:, :, -predicted.size(2) :]).mean()
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append(loss)
        return loss

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = pickle.load(open(self.hparams["data"], "rb"))
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()
        self.log("average_validation_loss", avg_loss)
        return avg_loss
