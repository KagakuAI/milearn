import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.nn import Linear, Sigmoid, ReLU, GELU, LeakyReLU, ELU, SiLU, Sequential
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import EarlyStopping
from .utils import TrainLogging, silence_and_seed_lightning
from .hopt import StepwiseHopt

def add_padding(x):
    bag_size = max(len(i) for i in x)
    mask = np.ones((len(x), bag_size, 1))

    out = []
    for i, bag in enumerate(x):
        bag = np.asarray(bag)
        if len(bag) < bag_size:
            mask[i][len(bag):] = 0
            padding = np.zeros((bag_size - bag.shape[0], bag.shape[1]))
            bag = np.vstack((bag, padding))
        out.append(bag)
    out_bags = np.asarray(out)
    return out_bags, mask

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
            y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1, 1)
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
        total_loss = nn.BCELoss(reduction='mean')(y_pred, y_true)
        return total_loss

    def prediction(self, out):
        out = Sigmoid()(out)
        return out


class BaseRegressor:
    def loss(self, y_pred, y_true):
        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true)
        return total_loss

    def prediction(self, out):
        return out


class FeatureExtractor:
    def __new__(cls, hidden_layer_sizes, activation="relu", dropout: float = 0.0, layer_norm: bool = False):

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leakyrelu": nn.LeakyReLU,
            "elu": nn.ELU,
            "silu": nn.SiLU,
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {list(activations.keys())}")

        act_fn = activations[activation]

        inp_dim = hidden_layer_sizes[0]
        net = []

        for dim in hidden_layer_sizes[1:]:
            net.append(nn.Linear(inp_dim, dim))

            if layer_norm:
                net.append(nn.LayerNorm(dim))

            net.append(act_fn())

            if dropout > 0.0:
                net.append(nn.Dropout(p=dropout))

            inp_dim = dim

        return nn.Sequential(*net)


class BaseNetwork(pl.LightningModule, StepwiseHopt):
    def __init__(self,
                 hidden_layer_sizes=(256, 128, 64),
                 max_epochs=100,
                 batch_size=128,
                 activation="gelu",
                 learning_rate=0.001,
                 early_stopping=True,
                 weight_decay=0.001,
                 dropout=0.0,
                 layer_norm=False,
                 num_workers=4,
                 verbose=False,
                 accelerator="cpu",
                 random_seed=42):
        super().__init__()
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.save_hyperparameters()
        silence_and_seed_lightning(seed=random_seed)

    def _create_basic_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):

        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes),
                                          activation=self.hparams.activation,
                                          dropout=self.hparams.dropout,
                                          layer_norm=self.hparams.layer_norm)
        self.estimator = nn.Linear(hidden_layer_sizes[-1], 1)

    def _create_specific_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y, m = batch
        b_embed, w_hat, y_hat = self.forward(x, m)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m = batch
        b_embed, w_hat, y_hat = self.forward(x, m)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, m = batch
        return self.forward(x, m)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
            )
        return optimizer

    def fit(self, x, y):

        # 1. Initialize network
        self._create_basic_layers(input_layer_size=x[0].shape[-1],
                                  hidden_layer_sizes=self.hparams.hidden_layer_sizes)
        self._create_specific_layers(input_layer_size=x[0].shape[-1],
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
        silence_and_seed_lightning(seed=self.hparams.random_seed)

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

    def _get_output(self, x):
        datamodule = DataModule(
            x,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )
        outputs = self._trainer.predict(self, datamodule=datamodule)
        return outputs, datamodule

    def predict(self, x):
        outputs, datamodule = self._get_output(x)
        y_pred = torch.cat([y for b, w, y in outputs], dim=0).cpu().numpy().flatten()
        return y_pred

    def get_bag_embedding(self, x):
        outputs, datamodule = self._get_output(x)
        bag_embed = torch.cat([b for b, w, y in outputs], dim=0).cpu().numpy()
        return bag_embed

    def get_instance_weights(self, x):
        outputs, datamodule = self._get_output(x)
        w_pred = torch.cat([w for b, w, y in outputs], dim=0)
        w_pred = [w[m.bool().squeeze(-1)].cpu().numpy() for w, m in zip(w_pred, datamodule.m)] # skip mask predicted weights
        return w_pred


