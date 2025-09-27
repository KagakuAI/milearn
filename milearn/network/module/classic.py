from .base import BaseNetwork, instance_dropout
from .hopt import StepwiseHopt
from typing import Any
from torch import Tensor
from typing import Tuple


class BagNetwork(BaseNetwork, StepwiseHopt):
    """A neural network model for multiple-instance learning (MIL) that
    aggregates instance embeddings into bag-level representations."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Initialize BagNetwork.

        Args:
            pool (str): pooling method, one of ['mean', 'sum', 'max', 'lse'].
            **kwargs: additional arguments for BaseNetwork.
        """
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, bags: Tensor, inst_mask: Tensor) -> Tensor:
        """Apply pooling over instance embeddings to create bag embeddings.

        Args:
            bags (torch.Tensor): instance embeddings.
            inst_mask (torch.Tensor): mask for valid instances.

        Returns:
            torch.Tensor: bag-level embeddings.
        """
        if self.pool == "mean":
            bag_embed = bags.sum(axis=1) / inst_mask.sum(axis=1)
        elif self.pool == "sum":
            bag_embed = bags.sum(axis=1)
        elif self.pool == "max":
            bag_embed = bags.max(dim=1)[0]
        elif self.pool == "lse":
            bag_embed = bags.exp().sum(dim=1).log()
        else:
            raise TypeError(f"Pooling type {self.pool} is not supported.")

        bag_embed = bag_embed.unsqueeze(1)
        return bag_embed

    def forward(self, bags: Tensor, inst_mask: Tensor) -> Tuple[Tensor, None, Tensor]:
        """Forward pass of BagNetwork.

        Args:
            bags (torch.Tensor): input bags of instances.
            inst_mask (torch.Tensor): instance mask.

        Returns:
            tuple: (bag embeddings, None, bag predictions).
        """
        inst_embed = self.instance_transformer(bags)
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed
        bag_embed = self._pooling(inst_embed, inst_mask)
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, None, bag_pred

    def hopt(self, x, y, param_grid, verbose=False):
        """Hyperparameter optimization with support for pooling methods.

        Args:
            x (list): input bags.
            y (list): labels.
            param_grid (dict): grid of hyperparameters.
            verbose (bool): verbosity flag.

        Returns:
            object: optimization results from StepwiseHopt.
        """
        valid_pools = ["mean", "sum", "max", "lse"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)


class InstanceNetwork(BaseNetwork):
    """A neural network model for multiple-instance learning (MIL) that
    aggregates predictions at the instance level."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Initialize InstanceNetwork.

        Args:
            pool (str): pooling method, one of ['mean', 'sum', 'max'].
            **kwargs: additional arguments for BaseNetwork.
        """
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, inst_pred: Tensor, inst_mask: Tensor) -> Tensor:
        """Apply pooling over instance predictions to create bag predictions.

        Args:
            inst_pred (torch.Tensor): predictions for instances.
            inst_mask (torch.Tensor): mask for valid instances.

        Returns:
            torch.Tensor: bag-level predictions.
        """
        if self.pool == "mean":
            bag_pred = inst_pred.sum(axis=1) / inst_mask.sum(axis=1)
        elif self.pool == "sum":
            bag_pred = inst_pred.sum(axis=1)
        elif self.pool == "max":
            idx = inst_pred.abs().argmax(dim=1, keepdim=True)
            bag_pred = inst_pred.gather(1, idx).squeeze(1)
        else:
            TypeError(f"Pooling type {self.pool} is not supported.")
            return None
        bag_pred = bag_pred.unsqueeze(1)
        return bag_pred

    def forward(self, bags: Tensor, inst_mask: Tensor) -> Tuple[None, None, Tensor]:
        """Forward pass of InstanceNetwork.

        Args:
            bags (torch.Tensor): input bags of instances.
            inst_mask (torch.Tensor): instance mask.

        Returns:
            tuple: (None, None, bag predictions).
        """
        inst_embed = self.instance_transformer(bags)
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed
        inst_score = self.bag_estimator(inst_embed)
        bag_score = self._pooling(inst_score, inst_mask)
        bag_pred = self.prediction(bag_score)

        return None, None, bag_pred

    def hopt(self, x, y, param_grid, verbose=True):
        """Hyperparameter optimization with support for pooling methods.

        Args:
            x (list): input bags.
            y (list): labels.
            param_grid (dict): grid of hyperparameters.
            verbose (bool): verbosity flag.

        Returns:
            object: optimization results from StepwiseHopt.
        """
        valid_pools = ["mean", "sum", "max"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)
