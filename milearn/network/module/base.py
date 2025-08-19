import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import Sigmoid, Linear, ReLU, Sequential
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
from .utils import add_padding, silence_lightning, TrainLogging


class DataModule(pl.LightningDataModule):
    def __init__(self, x, y=None, batch_size=32, num_workers=0, val_split=0.2):
        """
        x: input instances
        y: labels (optional, if None → inference mode)
        """
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        x, m = add_padding(self.x)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        m_tensor = torch.tensor(m, dtype=torch.float32)
        self.m = m_tensor

        if self.y is not None:
            y_tensor = torch.tensor(self.y, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, y_tensor, m_tensor)
            n_val = int(len(dataset) * self.val_split)
            seed = torch.Generator().manual_seed(42)
            self.train_ds, self.val_ds = random_split(dataset, [len(dataset)-n_val, n_val], generator=seed)
        else:
            self.dataset = TensorDataset(x_tensor, m_tensor)

    # Training/validation loaders
    def train_dataloader(self):
        if self.y is None:
            raise ValueError("No labels provided, cannot create train loader")
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.y is None:
            raise ValueError("No labels provided, cannot create val loader")
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    # Prediction loader
    def predict_dataloader(self):
        dataset = self.dataset

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class BaseClassifier:
    def loss(self, y_pred, y_true):
        total_loss = nn.BCELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss

    def get_pred(self, out):
        out = Sigmoid()(out)
        out = out.view(-1, 1)
        return out


class BaseRegressor:
    def loss(self, y_pred, y_true):
        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss

    def get_pred(self, out):
        out = out.view(-1, 1)
        return out


class FeatureExtractor:
    def __new__(cls, hidden_layer_sizes):
        inp_dim = hidden_layer_sizes[0]
        net = []
        for dim in hidden_layer_sizes[1:]:
            net.append(Linear(inp_dim, dim))
            net.append(ReLU())
            inp_dim = dim
        net = Sequential(*net)
        return net


class BaseNetwork(pl.LightningModule):
    def __init__(self,
                 hidden_layer_sizes=(256, 128, 64),
                 max_epochs=500,
                 batch_size=128,
                 learning_rate=0.001,
                 early_stopping=True,
                 weight_decay=0.001,
                 num_workers=1,
                 verbose=False,
                 accelerator="cpu"):
        super().__init__()
        self.save_hyperparameters()

        # Build a simple feed-forward backbone
        layers = []
        for in_dim, out_dim in zip(
                (hidden_layer_sizes[:-1]),
                (hidden_layer_sizes[1:])
        ):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))  # generic head
        self.model = nn.Sequential(*layers)

    def forward(self, x, m=None):
        return None, self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, m = batch
        w_hat, y_hat = self(x, m)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m = batch
        w_hat, y_hat = self(x, m)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, m = batch
        return self(x, m)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
            )
        return optimizer

    def fit(self, x, y):

        # 1. Initialize network
        input_layer_size = x[0].shape[-1] # TODO make consistent: x.shape[-1]
        self._initialize(input_layer_size=input_layer_size,
                         hidden_layer_sizes=self.hparams.hidden_layer_sizes)

        # 2. Prepare data
        datamodule = DataModule(x, y,
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                val_split=0.2)

        # 3. Trainer configuration
        callbacks = []
        if self.hparams.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor="val_loss", patience=10, mode="min"
            )
            callbacks.append(early_stop_callback)
        if self.hparams.verbose:
            logging_callback = TrainLogging()
            callbacks.append(logging_callback)

        silence_lightning()

        # 4. Build trainer
        self._trainer = pl.Trainer(
            max_epochs=self.hparams.max_epochs,
            callbacks=callbacks,
            accelerator=self.hparams.accelerator,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            deterministic=True,
        )
        # 5. Fit trainer
        self._trainer.fit(self, datamodule=datamodule)

        return self

    def predict(self, x):

        datamodule = DataModule(
            x,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

        outputs = self._trainer.predict(self, datamodule=datamodule)
        y_pred = torch.cat([y for w, y in outputs], dim=0).cpu().numpy().flatten()

        return y_pred


    def get_instance_weights(self, x):

        datamodule = DataModule(
            x,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

        outputs = self._trainer.predict(self, datamodule=datamodule)
        w_pred = torch.cat([w for w, y in outputs], dim=0)
        w_pred = w_pred.reshape(w_pred.shape[0], w_pred.shape[-1])
        w_pred = [np.asarray(i[j.bool().flatten()]) for i, j in zip(w_pred, datamodule.m)] # skip mask predicted weights

        return w_pred