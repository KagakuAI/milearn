import torch
import pytorch_lightning as pl
from torch.nn import Linear
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, random_split
from milearn.network.module.base import BaseNetwork, FeatureExtractor
from milearn.network.module.base import BaseRegressor, BaseClassifier
from milearn.network.module.utils import TrainLogging, silence_and_seed_lightning

class DataModule(pl.LightningDataModule):
    def __init__(self, x, y=None, batch_size=128, num_workers=0, val_split=0.2):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        if self.y is not None:
            y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)
            dataset = TensorDataset(x_tensor, y_tensor)
            n_val = int(len(dataset) * self.val_split)
            seed = torch.Generator().manual_seed(42)
            self.train_ds, self.val_ds = random_split(dataset, [len(dataset)-n_val, n_val], generator=seed)
        else:
            self.dataset = TensorDataset(x_tensor)

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

class MLPNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        silence_and_seed_lightning(seed=self.hparams.random_seed)

    def forward(self, X):

        # 1. Compute instance embeddings
        H = self.extractor(X)

        # 2. Compute final bag prediction
        y_score = self.estimator(H)
        y_pred = self.prediction(y_score)

        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        return self.forward(x)

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

    def predict(self, x):

        datamodule = DataModule(
            x,
            y=None,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

        outputs = self._trainer.predict(self, datamodule=datamodule)
        y_pred = torch.cat(outputs, dim=0).cpu().numpy().flatten()

        return y_pred

class MLPNetworkRegressor(MLPNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MLPNetworkClassifier(MLPNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



