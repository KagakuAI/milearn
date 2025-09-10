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

    def _create_special_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        self._create_attention(hidden_layer_sizes)

    def _create_attention(self, hidden_layer_sizes):
        raise NotImplementedError

    def forward(self, X, M):

        # 1. Compute instance embeddings
        inst_embed = self.instance_transformer(X)

        # 2. Apply instance dropout
        M = apply_instance_dropout(M, self.hparams.instance_dropout, self.training)
        inst_embed = M * inst_embed

        # 3. Compute instance attention weights
        bag_embed, inst_weights = self.compute_attention(inst_embed, M)

        # 4. Compute final bag prediction
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, inst_weights, bag_pred

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
        inst_logits = self.attention(H) / self.tau

        # 2. Mask padded instances
        mask_bool = M.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))

        # 3. Compute weights
        inst_weights = torch.softmax(inst_logits, dim=1)

        # 4. Weighted sum to get bag embedding
        bag_embed = torch.sum(inst_weights * H, dim=1, keepdim=True)

        return bag_embed, inst_weights

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
        inst_logits = torch.matmul(Q, K.transpose(1, 2)) / (self.tau * (H.shape[-1] ** 0.5))

        # 3. Mask invalid instances
        mask_bool = M.squeeze(-1).bool()
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
        super().__init__(**kwargs)
        self.beta = tau

    def _create_attention(self, hidden_layer_sizes):
        self.query_vector = nn.Parameter(torch.randn(1, hidden_layer_sizes[-1]))

    def compute_attention(self, H, M):

        B, N, D = H.shape

        # 1. Expand query vector to batch
        q = self.query_vector.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        # 2. Compute scores
        inst_logits = self.beta * torch.bmm(q, H.transpose(1, 2))  # [B, 1, N]

        # 3. Mask invalid instances
        mask_bool = M.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # 4. Attention weights
        inst_weights = torch.softmax(inst_logits, dim=-1)
        inst_weights = inst_weights.transpose(1, 2)

        # 5. Compute bag embedding
        bag_embed = torch.bmm(inst_weights.transpose(1, 2), H)  # [B, 1, D]

        return bag_embed, inst_weights
