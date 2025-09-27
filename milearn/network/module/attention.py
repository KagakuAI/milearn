import torch
from torch import nn
from .base import BaseNetwork, instance_dropout

class BaseAttentionNetwork(BaseNetwork):
    def __init__(self, tau=1.0, **kwargs):
        """
        Base class for attention-based MIL networks.

        Args:
            tau (float): temperature scaling parameter for attention
            **kwargs: additional arguments passed to BaseNetwork
        """
        super().__init__(**kwargs)
        self.tau = tau

    def _create_special_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        """
        Create attention layers for the network.

        Args:
            input_layer_size (int): size of the input features
            hidden_layer_sizes (tuple[int, ...]): sizes of hidden layers
        """
        self._create_attention(hidden_layer_sizes)

    def _create_attention(self, hidden_layer_sizes):
        """
        Define the attention mechanism.

        Args:
            hidden_layer_sizes (tuple[int, ...]): sizes of hidden layers

        Raises:
            NotImplementedError: must be implemented in subclasses
        """
        raise NotImplementedError

    def forward(self, bags, inst_mask):
        """
        Forward pass through the attention network.

        Args:
            bags (torch.Tensor): batch of input bags (B, N, D)
            inst_mask (torch.Tensor): instance mask (B, N, 1)

        Returns:
            tuple: (bag_embed, inst_weights, bag_pred)
                - bag_embed (torch.Tensor): bag-level embedding
                - inst_weights (torch.Tensor): attention weights per instance
                - bag_pred (torch.Tensor): final bag prediction
        """
        # 1. Compute instance embeddings
        inst_embed = self.instance_transformer(bags)

        # 2. Apply instance dropout
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed

        # 3. Compute instance attention weights
        bag_embed, inst_weights = self.compute_attention(inst_embed, inst_mask)

        # 4. Compute final bag prediction
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, inst_weights, bag_pred

    def compute_attention(self, H, M):
        """
        Compute attention weights and bag embedding.

        Args:
            H (torch.Tensor): instance embeddings (B, N, D)
            M (torch.Tensor): instance mask (B, N, 1)

        Raises:
            NotImplementedError: must be implemented in subclasses
        """
        raise NotImplementedError


class AdditiveAttentionNetwork(BaseAttentionNetwork):
    def _create_attention(self, hidden_layer_sizes):
        """
        Create additive attention mechanism.

        Args:
            hidden_layer_sizes (tuple[int, ...]): sizes of hidden layers
        """
        self.attention = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
            nn.Tanh(),
            nn.Linear(hidden_layer_sizes[-1], 1)
        )

    def compute_attention(self, inst_embed, inst_mask):
        """
        Compute additive attention.

        Args:
            inst_embed (torch.Tensor): instance embeddings (B, N, D)
            inst_mask (torch.Tensor): instance mask (B, N, 1)

        Returns:
            tuple: (bag_embed, inst_weights)
                - bag_embed (torch.Tensor): bag-level embedding
                - inst_weights (torch.Tensor): attention weights per instance
        """
        # 1. Compute logits
        inst_logits = self.attention(inst_embed) / self.tau

        # 2. Mask padded instances
        mask_bool = inst_mask.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))

        # 3. Compute weights
        inst_weights = torch.softmax(inst_logits, dim=1)

        # 4. Weighted sum to get bag embedding
        bag_embed = torch.sum(inst_weights * inst_embed, dim=1, keepdim=True)

        return bag_embed, inst_weights


class SelfAttentionNetwork(BaseAttentionNetwork):
    def _create_attention(self, hidden_layer_sizes):
        """
        Create self-attention mechanism.

        Args:
            hidden_layer_sizes (tuple[int, ...]): sizes of hidden layers
        """
        D = hidden_layer_sizes[-1]
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)

    def compute_attention(self, inst_embed, inst_mask):
        """
        Compute self-attention.

        Args:
            inst_embed (torch.Tensor): instance embeddings (B, N, D)
            inst_mask (torch.Tensor): instance mask (B, N, 1)

        Returns:
            tuple: (bag_embed, inst_weights)
                - bag_embed (torch.Tensor): bag-level embedding
                - inst_weights (torch.Tensor): attention weights per instance
        """
        # 1. Project to Q, K, V
        Q = self.q_proj(inst_embed)
        K = self.k_proj(inst_embed)
        V = self.v_proj(inst_embed)

        # 2. Compute scaled dot-product attention
        inst_logits = torch.matmul(Q, K.transpose(1, 2)) / (self.tau * (inst_embed.shape[-1] ** 0.5))

        # 3. Mask invalid instances
        mask_bool = inst_mask.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # 4. Compute attention weights
        inst_weights = torch.softmax(inst_logits, dim=-1) # (B, N, N)

        # 5. Reduce to per-instance / Incoming (who gets attended to)
        inst_weights = inst_weights.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, N, 1)

        # 6. Weighted sum of values -> bag embedding
        bag_embed = torch.sum(inst_weights * V, dim=1, keepdim=True)  # (B, 1, D)

        return bag_embed, inst_weights


class HopfieldAttentionNetwork(BaseAttentionNetwork):
    def __init__(self, tau=1.0, **kwargs):
        """
        Hopfield-style attention network.

        Args:
            tau (float): scaling factor for attention (used as beta)
            **kwargs: additional arguments passed to BaseNetwork
        """
        super().__init__(**kwargs)
        self.beta = tau

    def _create_attention(self, hidden_layer_sizes):
        """
        Create Hopfield-style attention mechanism.

        Args:
            hidden_layer_sizes (tuple[int, ...]): sizes of hidden layers
        """
        self.query_vector = nn.Parameter(torch.randn(1, hidden_layer_sizes[-1]))

    def compute_attention(self, inst_embed, inst_mask):
        """
        Compute Hopfield-style attention.

        Args:
            inst_embed (torch.Tensor): instance embeddings (B, N, D)
            inst_mask (torch.Tensor): instance mask (B, N, 1)

        Returns:
            tuple: (bag_embed, inst_weights)
                - bag_embed (torch.Tensor): bag-level embedding
                - inst_weights (torch.Tensor): attention weights per instance
        """
        B, N, D = inst_embed.shape

        # 1. Expand query vector to batch
        q = self.query_vector.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        # 2. Compute scores
        inst_logits = self.beta * torch.bmm(q, inst_embed.transpose(1, 2))  # [B, 1, N]

        # 3. Mask invalid instances
        mask_bool = inst_mask.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # 4. Attention weights
        inst_weights = torch.softmax(inst_logits, dim=-1)
        inst_weights = inst_weights.transpose(1, 2)

        # 5. Compute bag embedding
        bag_embed = torch.bmm(inst_weights.transpose(1, 2), inst_embed)  # [B, 1, D]

        return bag_embed, inst_weights
