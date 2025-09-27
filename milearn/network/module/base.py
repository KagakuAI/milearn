
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader, TensorDataset, random_split

from .hopt import StepwiseHopt
from .utils import TrainLogging, silence_and_seed_lightning
from numpy import ndarray
from typing import List
from typing import Union
from typing import Tuple
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from torch.nn.modules.activation import GELU
from torch.nn.modules.linear import Linear
from milearn.network.module.base import DataModule
from milearn.network.module.mlp import DataModule
from typing import Any
from typing import Optional
from torch.optim.adamw import AdamW


def instance_dropout(m: Tensor, p: float = 0.0, training: bool = True) -> Tensor:
    """Apply instance-level dropout with optional mask fixing.

    Args:
        m (torch.Tensor): instance mask (B, N)
        p (float): dropout probability
        training (bool): whether in training mode

    Returns:
        torch.Tensor: updated mask with dropout applied
    """
    if training and p > 0.0:
        # Drop only real instances
        drop_mask = torch.ones_like(m, dtype=torch.float)
        real_mask = (m > 0).float()

        rand_vals = torch.rand_like(m.float())
        drop_mask = ((rand_vals > p) | (m == 0)).float()  # keep padded = 0, drop real with prob p

        # Ensure at least one real survives per bag
        real_counts = (real_mask * drop_mask).sum(dim=1, keepdim=True)
        needs_fix = real_counts == 0
        if needs_fix.any():
            # For each bad bag, pick one real instance to restore
            for i in torch.where(needs_fix.squeeze())[0]:
                real_indices = torch.where(real_mask[i] > 0)[0]
                j = real_indices[torch.randint(len(real_indices), (1,))]
                drop_mask[i, j] = 1.0

        m = m * drop_mask
    return m


class DataModule(pl.LightningDataModule):
    def __init__(self, x: List[ndarray], y: Union[List[float], List[int], None] = None, batch_size: int = 128, num_workers: int = 0, val_split: float = 0.2) -> None:
        """Data module for bagged input with padding and splitting.

        Args:
            x (list[np.ndarray]): list of bags, each bag is an array of instances
            y (np.ndarray | None): labels for each bag
            batch_size (int): number of bags per batch
            num_workers (int): number of workers for dataloaders
            val_split (float): validation split ratio
        """
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def add_padding(self, x: List[ndarray]) -> Tuple[ndarray, ndarray]:
        """Pad bags to the same size and create mask.

        Args:
            x (list[np.ndarray]): list of bags

        Returns:
            tuple:
                - out_bags (np.ndarray): padded bags
                - mask (np.ndarray): mask with 1 for real, 0 for padding
        """
        bag_size = max(len(i) for i in x)
        mask = np.ones((len(x), bag_size, 1))

        out = []
        for i, bag in enumerate(x):
            bag = np.asarray(bag)
            if len(bag) < bag_size:
                mask[i][len(bag) :] = 0
                padding = np.zeros((bag_size - bag.shape[0], bag.shape[1]))
                bag = np.vstack((bag, padding))
            out.append(bag)
        out_bags = np.asarray(out)
        return out_bags, mask

    def setup(self, stage: TrainerFn = None) -> None:
        """Prepare datasets with optional train/val split.

        Args:
            stage (str | None): stage name ("fit", "validate", "predict", etc.)
        """
        x, m = self.add_padding(self.x)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        m_tensor = torch.tensor(m, dtype=torch.float32)
        self.m = m_tensor

        if self.y is not None:
            y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1, 1)
            dataset = TensorDataset(x_tensor, y_tensor, m_tensor)
            n_val = int(len(dataset) * self.val_split)
            seed = torch.Generator().manual_seed(42)
            self.train_ds, self.val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=seed)
        else:
            self.dataset = TensorDataset(x_tensor, m_tensor)

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader.

        Returns:
            DataLoader: training dataloader

        Raises:
            ValueError: if labels are not provided
        """
        if self.y is None:
            raise ValueError("No labels provided, cannot create train loader")
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader.

        Returns:
            DataLoader: validation dataloader

        Raises:
            ValueError: if labels are not provided
        """
        if self.y is None:
            raise ValueError("No labels provided, cannot create val loader")
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """Get prediction dataloader.

        Returns:
            DataLoader: prediction dataloader
        """
        dataset = self.dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class BaseClassifier:
    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute binary cross-entropy loss.

        Args:
            y_pred (torch.Tensor): predicted values
            y_true (torch.Tensor): ground-truth labels

        Returns:
            torch.Tensor: BCE loss
        """
        total_loss = nn.BCELoss(reduction="mean")(y_pred, y_true)
        return total_loss

    def prediction(self, out: Tensor) -> Tensor:
        """Apply sigmoid to raw outputs.

        Args:
            out (torch.Tensor): raw outputs

        Returns:
            torch.Tensor: probabilities
        """
        out = Sigmoid()(out)
        return out


class BaseRegressor:
    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute mean squared error loss.

        Args:
            y_pred (torch.Tensor): predicted values
            y_true (torch.Tensor): ground-truth values

        Returns:
            torch.Tensor: MSE loss
        """
        total_loss = nn.MSELoss(reduction="mean")(y_pred, y_true)
        return total_loss

    def prediction(self, out: Tensor) -> Tensor:
        """Identity prediction for regression.

        Args:
            out (torch.Tensor): raw outputs

        Returns:
            torch.Tensor: same as input
        """
        return out


class InstanceTransformer:
    def __new__(cls, hidden_layer_sizes, activation="relu"):
        """Create a feed-forward transformer for instances.

        Args:
            hidden_layer_sizes (tuple[int, ...]): layer sizes (input_dim, hidden_dims...)
            activation (str): activation function name

        Returns:
            nn.Sequential: transformer network
        """
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
            net.append(act_fn())
            inp_dim = dim

        return nn.Sequential(*net)


class BaseNetwork(pl.LightningModule, StepwiseHopt):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, int, int] = (256, 128, 64),
        max_epochs: int = 1000,
        batch_size: int = 128,
        activation: str = "gelu",
        learning_rate: float = 0.001,
        early_stopping: bool = True,
        weight_decay: float = 0.0,
        instance_dropout: float = 0.0,
        accelerator: str = "cpu",
        verbose: bool = False,
        random_seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        """Base network class with training utilities.

        Args:
            hidden_layer_sizes (tuple[int, ...]): hidden layer sizes
            max_epochs (int): maximum training epochs
            batch_size (int): training batch size
            activation (str): activation function
            learning_rate (float): optimizer learning rate
            early_stopping (bool): whether to enable early stopping
            weight_decay (float): L2 regularization coefficient
            instance_dropout (float): dropout probability for instances
            accelerator (str): training accelerator ("cpu" or "gpu")
            verbose (bool): enable training logging
            random_seed (int): random seed for reproducibility
            num_workers (int): number of workers for dataloaders
        """
        super().__init__()
        self.random_seed = random_seed
        self.save_hyperparameters()
        silence_and_seed_lightning(seed=self.random_seed)

    def _init_weights(self, m: Union[GELU, Linear]) -> None:
        """Initialize weights deterministically.

        Args:
            m (nn.Module): module to initialize
        """
        if isinstance(m, nn.Linear):
            # deterministic Xavier uniform initialization
            torch.manual_seed(self.random_seed)
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def _create_basic_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        """Create instance transformer and bag estimator.

        Args:
            input_layer_size (int): input dimension
            hidden_layer_sizes (tuple[int, ...]): hidden layer sizes
        """
        self.instance_transformer = InstanceTransformer(
            (input_layer_size, *hidden_layer_sizes), activation=self.hparams.activation
        )
        self.bag_estimator = nn.Linear(hidden_layer_sizes[-1], 1)

    def _create_special_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        """Placeholder for special layers in subclasses.

        Args:
            input_layer_size (int): input dimension
            hidden_layer_sizes (tuple[int, ...]): hidden layer sizes

        Returns:
            NotImplementedError
        """
        return NotImplementedError

    def _create_and_fit_trainer(self, datamodule: Union[DataModule, DataModule]) -> Optional[Any]:
        """Create and fit a PyTorch Lightning trainer.

        Args:
            datamodule (DataModule): datamodule for training
        """
        # 1. Trainer configuration
        callbacks = []
        if self.hparams.early_stopping:
            early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
            callbacks.append(early_stop_callback)
        if self.hparams.verbose:
            logging_callback = TrainLogging()
            callbacks.append(logging_callback)
        silence_and_seed_lightning(seed=self.hparams.random_seed)

        # 2. Build trainer
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

        # 3. Fit trainer
        self._trainer.fit(self, datamodule=datamodule)

        return None

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch (tuple): (x, y, mask)
            batch_idx (int): batch index

        Returns:
            torch.Tensor: training loss
        """
        x, y, m = batch
        b_embed, w_hat, y_hat = self.forward(x, m)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Validation step.

        Args:
            batch (tuple): (x, y, mask)
            batch_idx (int): batch index

        Returns:
            torch.Tensor: validation loss
        """
        x, y, m = batch
        b_embed, w_hat, y_hat = self.forward(x, m)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: List[Tensor], batch_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        """Prediction step.

        Args:
            batch (tuple): (x, mask)
            batch_idx (int): batch index

        Returns:
            tuple: model outputs
        """
        x, m = batch
        return self.forward(x, m)

    def configure_optimizers(self) -> AdamW:
        """Configure optimizer.

        Returns:
            torch.optim.Optimizer: AdamW optimizer
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def fit(self, x: List[ndarray], y: Union[List[float], List[int]]) -> Any:
        """Fit the model.

        Args:
            x (list[np.ndarray]): input bags
            y (np.ndarray): labels

        Returns:
            BaseNetwork: self
        """
        silence_and_seed_lightning(seed=self.random_seed)

        # 1. Initialize network
        self._create_basic_layers(input_layer_size=x[0].shape[-1], hidden_layer_sizes=self.hparams.hidden_layer_sizes)
        self._create_special_layers(input_layer_size=x[0].shape[-1], hidden_layer_sizes=self.hparams.hidden_layer_sizes)
        self.apply(self._init_weights)

        # 2. Prepare data
        datamodule = DataModule(
            x, y, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, val_split=0.2
        )

        # 3. Create and fit trainer
        self._create_and_fit_trainer(datamodule)

        return self

    def _get_output(self, x: List[ndarray]) -> Tuple[Union[List[Tuple[None, None, Tensor]], List[Tuple[Tensor, None, Tensor]], List[Tuple[Tensor, Tensor, Tensor]]], DataModule]:
        """Run prediction with datamodule.

        Args:
            x (list[np.ndarray]): input bags

        Returns:
            tuple: (outputs, datamodule)
        """
        datamodule = DataModule(x, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        outputs = self._trainer.predict(self, datamodule=datamodule)
        return outputs, datamodule

    def predict(self, x: List[ndarray]) -> ndarray:
        """Predict bag-level outputs.

        Args:
            x (list[np.ndarray]): input bags

        Returns:
            np.ndarray: predictions
        """
        outputs, datamodule = self._get_output(x)
        y_pred = torch.cat([y for b, w, y in outputs], dim=0).cpu().numpy().flatten()
        return y_pred

    def get_bag_embedding(self, x):
        """Get bag embeddings.

        Args:
            x (list[np.ndarray]): input bags

        Returns:
            np.ndarray: bag embeddings
        """
        outputs, datamodule = self._get_output(x)
        bag_embed = torch.cat([b for b, w, y in outputs], dim=0).cpu().numpy()
        return bag_embed

    def get_instance_weights(self, x):
        """Get instance attention weights.

        Args:
            x (list[np.ndarray]): input bags

        Returns:
            list[np.ndarray]: instance weights per bag
        """
        outputs, datamodule = self._get_output(x)
        w_pred = torch.cat([w for b, w, y in outputs], dim=0)
        w_pred = [
            w[m.bool().squeeze(-1)].cpu().numpy() for w, m in zip(w_pred, datamodule.m)
        ]  # skip mask predicted weights
        return w_pred
