import torch
from torch import nn
from .base import BaseNetwork, apply_instance_dropout

class BaseAttentionNetwork(BaseNetwork):
    def __init__(self, tau=1.0, instance_dropout=0.0, **kwargs):
        """
        Base class for attention-based MIL networks
        """
        super().__init__(**kwargs)
        self.tau = tau
        self.instance_dropout = instance_dropout

    def _create_special_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        self._create_attention(hidden_layer_sizes)

    def _create_attention(self, hidden_layer_sizes):
        raise NotImplementedError

    def _weight_dropout(self, weights, p=0.0, training=True):

        if training and p > 0.0:
            drop_mask = (torch.rand_like(weights) > p).float()
            weights = weights * drop_mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights


    def forward(self, X, M):

        # 1. Compute instance embeddings
        H = self.extractor(X)

        # 2. Apply instance dropout
        M = apply_instance_dropout(M, self.instance_dropout, self.training)
        H = M * H

        # 3. Compute instance attention weights
        bag_embed, weights = self.compute_attention(H, M)

        # 4. Compute final bag prediction
        bag_score = self.estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, weights, bag_pred

    def compute_attention(self, H, M):
        raise NotImplementedError

class AdditiveAttentionNetwork(BaseAttentionNetwork):
    def _create_attention(self, hidden_layer_sizes):

        self.attention = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
            nn.Tanh(),
            nn.Linear(hidden_layer_sizes[-1], 1)
        )

    def compute_attention(self, H, M):

        # 1. Compute logits
        logits = self.attention(H) / self.tau

        # 2. Mask padded instances
        mask_bool = M.squeeze(-1).bool()
        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))

        # 3. Compute weights
        weights = torch.softmax(logits, dim=1)

        # 4. Weighted sum to get bag embedding
        bag_embed = torch.sum(weights * H, dim=1, keepdim=True)

        return bag_embed, weights

class SelfAttentionNetwork(BaseAttentionNetwork):
    def _create_attention(self, hidden_layer_sizes):

        D = hidden_layer_sizes[-1]
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)

    def compute_attention(self, H, M):

        # 1. Project to Q, K, V
        Q = self.q_proj(H)
        K = self.k_proj(H)
        V = self.v_proj(H)

        # 2. Compute scaled dot-product attention
        logits = torch.matmul(Q, K.transpose(1, 2)) / (self.tau * (H.shape[-1] ** 0.5))

        # 3. Mask invalid instances
        mask_bool = M.squeeze(-1).bool()
        logits = logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # 4. Compute attention weights
        weights = torch.softmax(logits, dim=-1) # (B, N, N)

        # 5. Reduce to per-instance / Incoming (who gets attended to)
        weights = weights.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, N, 1)

        # 6. Weighted sum of values -> bag embedding
        bag_embed = torch.sum(weights * V, dim=1, keepdim=True)  # (B, 1, D)

        return bag_embed, weights

class HopfieldAttentionNetwork(BaseAttentionNetwork):
    def __init__(self, tau=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = tau

    def _create_attention(self, hidden_layer_sizes):
        self.query_vector = nn.Parameter(torch.randn(1, hidden_layer_sizes[-1]))

    def compute_attention(self, H, M):

        B, N, D = H.shape

        # 1. Expand query vector to batch
        q = self.query_vector.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        # 2. Compute scores
        logits = self.beta * torch.bmm(q, H.transpose(1, 2))  # [B, 1, N]

        # 3. Mask invalid instances
        mask_bool = M.squeeze(-1).bool()
        logits = logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # 4. Attention weights
        weights = torch.softmax(logits, dim=-1)
        weights = weights.transpose(1, 2)

        # 5. Compute bag embedding
        bag_embed = torch.bmm(weights.transpose(1, 2), H)  # [B, 1, D]

        return bag_embed, weights
