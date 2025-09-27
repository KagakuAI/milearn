from .module.attention import AdditiveAttentionNetwork, HopfieldAttentionNetwork, SelfAttentionNetwork
from .module.base import BaseClassifier
from .module.classic import BagNetwork, InstanceNetwork
from .module.dynamic import DynamicPoolingNetwork, MarginLoss
from .module.mlp import BagWrapperMLPNetwork, InstanceWrapperMLPNetwork


class BagNetworkClassifier(BagNetwork, BaseClassifier):
    """Bag-level network with mean/sum/max pooling for classification."""

    def __init__(self, pool="mean", **kwargs):
        """Initialize BagNetworkClassifier.

        Args:
            pool (str): pooling strategy ("mean", "sum", "max", "lse").
            **kwargs: additional arguments for BagNetwork.
        """
        super().__init__(pool=pool, **kwargs)


class InstanceNetworkClassifier(InstanceNetwork, BaseClassifier):
    """Instance-level network with per-instance predictions pooled to bag-level
    for classification."""

    def __init__(self, pool="mean", **kwargs):
        """Initialize InstanceNetworkClassifier.

        Args:
            pool (str): pooling strategy ("mean", "sum", "max").
            **kwargs: additional arguments for InstanceNetwork.
        """
        super().__init__(pool=pool, **kwargs)


class AdditiveAttentionNetworkClassifier(AdditiveAttentionNetwork, BaseClassifier):
    """Additive attention network adapted for classification."""

    def __init__(self, **kwargs):
        """Initialize AdditiveAttentionNetworkClassifier.

        Args:
            **kwargs: additional arguments for AdditiveAttentionNetwork.
        """
        super().__init__(**kwargs)


class SelfAttentionNetworkClassifier(SelfAttentionNetwork, BaseClassifier):
    """Self-attention network adapted for classification."""

    def __init__(self, **kwargs):
        """Initialize SelfAttentionNetworkClassifier.

        Args:
            **kwargs: additional arguments for SelfAttentionNetwork.
        """
        super().__init__(**kwargs)


class HopfieldAttentionNetworkClassifier(HopfieldAttentionNetwork, BaseClassifier):
    """Hopfield-style attention network adapted for classification."""

    def __init__(self, **kwargs):
        """Initialize HopfieldAttentionNetworkClassifier.

        Args:
            **kwargs: additional arguments for HopfieldAttentionNetwork.
        """
        super().__init__(**kwargs)


class BagWrapperMLPNetworkClassifier(BagWrapperMLPNetwork, BaseClassifier):
    """MLP network with bag-level pooling for classification."""

    def __init__(self, **kwargs):
        """Initialize BagWrapperMLPNetworkClassifier.

        Args:
            **kwargs: additional arguments for BagWrapperMLPNetwork.
        """
        super().__init__(**kwargs)


class InstanceWrapperMLPNetworkClassifier(InstanceWrapperMLPNetwork, BaseClassifier):
    """MLP network with instance-level predictions pooled to bag-level for
    classification."""

    def __init__(self, **kwargs):
        """Initialize InstanceWrapperMLPNetworkClassifier.

        Args:
            **kwargs: additional arguments for InstanceWrapperMLPNetwork.
        """
        super().__init__(**kwargs)


class DynamicPoolingNetworkClassifier(DynamicPoolingNetwork, BaseClassifier):
    """Dynamic pooling network adapted for classification."""

    def __init__(self, **kwargs):
        """Initialize DynamicPoolingNetworkClassifier.

        Args:
            **kwargs: additional arguments for DynamicPoolingNetwork.
        """
        super().__init__(**kwargs)

    def loss(self, y_pred, y_true):
        """Compute margin-based loss for classification.

        Args:
            y_pred (torch.Tensor): predicted bag embeddings.
            y_true (torch.Tensor): true labels.

        Returns:
            torch.Tensor: computed margin loss.
        """
        margin_loss = MarginLoss()
        loss = margin_loss(y_pred, y_true)
        return loss
