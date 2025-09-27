import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .module.attention import AdditiveAttentionNetwork, HopfieldAttentionNetwork, SelfAttentionNetwork
from .module.base import BaseRegressor
from .module.classic import BagNetwork, InstanceNetwork
from .module.dynamic import DynamicPoolingNetwork
from .module.mlp import BagWrapperMLPNetwork, InstanceWrapperMLPNetwork
from typing import Any
from numpy import ndarray
from typing import List


class BagNetworkRegressor(BagNetwork, BaseRegressor):
    """Bag-level network with mean/sum/max pooling for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize BagNetworkRegressor.

        Args:
            **kwargs: additional arguments for BagNetwork.
        """
        super().__init__(**kwargs)


class InstanceNetworkRegressor(InstanceNetwork, BaseRegressor):
    """Instance-level network with per-instance predictions pooled to bag-level
    for regression."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize InstanceNetworkRegressor.

        Args:
            **kwargs: additional arguments for InstanceNetwork.
        """
        super().__init__(**kwargs)


class AdditiveAttentionNetworkRegressor(AdditiveAttentionNetwork, BaseRegressor):
    """Additive attention network adapted for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize AdditiveAttentionNetworkRegressor.

        Args:
            **kwargs: additional arguments for AdditiveAttentionNetwork.
        """
        super().__init__(**kwargs)


class SelfAttentionNetworkRegressor(SelfAttentionNetwork, BaseRegressor):
    """Self-attention network adapted for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SelfAttentionNetworkRegressor.

        Args:
            **kwargs: additional arguments for SelfAttentionNetwork.
        """
        super().__init__(**kwargs)


class HopfieldAttentionNetworkRegressor(HopfieldAttentionNetwork, BaseRegressor):
    """Hopfield-style attention network adapted for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize HopfieldAttentionNetworkRegressor.

        Args:
            **kwargs: additional arguments for HopfieldAttentionNetwork.
        """
        super().__init__(**kwargs)


class BagWrapperMLPNetworkRegressor(BagWrapperMLPNetwork, BaseRegressor):
    """MLP network with bag-level pooling for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize BagWrapperMLPNetworkRegressor.

        Args:
            **kwargs: additional arguments for BagWrapperMLPNetwork.
        """
        super().__init__(**kwargs)


class InstanceWrapperMLPNetworkRegressor(InstanceWrapperMLPNetwork, BaseRegressor):
    """MLP network with instance-level predictions pooled to bag-level for
    regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize InstanceWrapperMLPNetworkRegressor.

        Args:
            **kwargs: additional arguments for InstanceWrapperMLPNetwork.
        """
        super().__init__(**kwargs)


class DynamicPoolingNetworkRegressor(DynamicPoolingNetwork, BaseRegressor):
    """Dynamic pooling network adapted for regression tasks.

    Performs Min-Max scaling on target values during training and
    inverse transforms predictions.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize DynamicPoolingNetworkRegressor.

        Args:
            **kwargs: additional arguments for DynamicPoolingNetwork.
        """
        super().__init__(**kwargs)

    def fit(self, x: List[ndarray], y: List[float]) -> "DynamicPoolingNetworkRegressor":
        """Fit the network on training data with scaled target values.

        Args:
            x (list or array-like): Input bags.
            y (list or array-like): Target values.

        Returns:
            self: fitted network instance.
        """
        y = np.array(y).reshape(-1, 1)
        self.scaler = MinMaxScaler()
        y = self.scaler.fit_transform(y).flatten()

        return super().fit(x, y)

    def predict(self, x: List[ndarray]) -> ndarray:
        """Predict target values for input bags and inverse transform scaling.

        Args:
            x (list or array-like): Input bags.

        Returns:
            np.ndarray: predicted target values, scaled back to original range.
        """
        y_pred = super().predict(x)
        y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred
