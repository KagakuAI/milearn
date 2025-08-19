import torch
from torch import nn
from .base import BaseNetwork, FeatureExtractor


class AttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        """
        Initialize layers:
        - Feature extractor
        - Detector network producing scalar attention logits per instance
        - Estimator mapping aggregated features to output
        """
        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.detector = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
            nn.Tanh(),
            nn.Linear(hidden_layer_sizes[-1], 1)
        )
        self.estimator = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        Args:
            x: Input instances (B, N, D_in)
            m: Mask for valid instances (B, N, 1)

        Returns:
            weights: Attention weights (B, 1, N)
            out: Bag-level prediction (B, 1, 1)
        """
        x_feat = self.extractor(x)  # (B, N, D_hidden)
        logits = self.detector(x_feat)  # (B, N, 1)
        logits = logits.transpose(2, 1)          # (B, 1, N)

        masked_logits = logits.masked_fill(m.transpose(2, 1) == 0, float('-inf'))
        weights = torch.softmax(masked_logits, dim=2)   # (B, 1, N)

        bag_embedding = torch.bmm(weights, x_feat)  # (B, 1, D_hidden)
        bag_score = self.estimator(bag_embedding)          # (B, 1, 1)
        bag_pred = self.get_pred(bag_score)

        return weights, bag_pred

class TempAttentionNetwork(AttentionNetwork):
    def __init__(self, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x_feat = self.extractor(x)
        logits = m * self.detector(x_feat)
        logits = logits.transpose(2, 1)

        masked_logits = logits.masked_fill(m.transpose(2, 1) == 0, float('-inf'))
        weights = torch.softmax(masked_logits / self.tau, dim=2)

        bag_embedding = torch.bmm(weights, x_feat)
        bag_score = self.estimator(bag_embedding)
        bag_pred = self.get_pred(bag_score)

        return weights, bag_pred

class GatedAttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))

        self.gate_V = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
            nn.Tanh()
        )
        self.gate_U = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
            nn.Sigmoid()
        )
        self.detector = nn.Linear(hidden_layer_sizes[-1], 1)
        self.estimator = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        Args:
            x: (B, N, D_in)
            m: (B, N, 1)

        Returns:
            weights: Attention weights (B, 1, N)
            out: Bag-level prediction (B, 1, 1)
        """
        x_feat = self.extractor(x)  # (B, N, D_hidden)
        x_gated = self.gate_V(x_feat) * self.gate_U(x_feat) # (B, N, D_hidden)

        logits = self.detector(x_gated)  # (B, N, 1)
        logits = logits.transpose(2, 1)  # (B, 1, N)

        masked_logits = logits.masked_fill(m.transpose(2, 1) == 0, float('-inf'))
        weights = torch.softmax(masked_logits, dim=2)  # (B, 1, N)

        bag_embedding = torch.bmm(weights, x_feat)  # (B, 1, D_hidden)
        bag_score = self.estimator(bag_embedding)          # (B, 1, 1)
        bag_pred = self.get_pred(bag_score)

        return weights, bag_pred

class MultiHeadAttentionNetwork(BaseNetwork):
    def __init__(self, num_heads: int = 2, **kwargs):
        self.num_heads = num_heads
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.detector = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
                nn.Tanh(),
                nn.Linear(hidden_layer_sizes[-1], 1)
            )
            for _ in range(self.num_heads)
        ])
        self.estimator = nn.Linear(hidden_layer_sizes[-1] * self.num_heads, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        Args:
            x: (B, N, D_in)
            m: (B, N, 1)

        Returns:
            avg_weights: Average attention weights across heads (B, 1, N)
            out: Bag-level prediction (B, 1, 1)
        """
        x_feat = self.extractor(x)  # (B, N, D_hidden)

        head_outputs, head_weights = [], []
        for head in self.detector:

            logits = head(x_feat)  # (B, N, 1)

            masked_logits = logits.masked_fill(m == 0, float('-inf'))
            weights = torch.softmax(masked_logits, dim=1)  # (B, N, 1)

            head_embedding = torch.sum(weights * x_feat, dim=1)  # (B, D_hidden)
            head_outputs.append(head_embedding)
            head_weights.append(weights)

        weights = torch.stack(head_weights, dim=0).mean(dim=0).transpose(2, 1)  # (B, 1, N)
        bag_embedding = torch.cat(head_outputs, dim=1)  # (B, D_hidden * num_heads)

        bag_score = self.estimator(bag_embedding)              # (B, 1)
        bag_pred = self.get_pred(bag_score.unsqueeze(1).unsqueeze(2))  # reshape (B, 1, 1)

        return weights, bag_pred


class SelfAttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):

        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))

        self.w_query = nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1])
        self.w_key = nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1])
        self.w_value = nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1])

        self.detector = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
            nn.Tanh(),
            nn.Linear(hidden_layer_sizes[-1], 1)
        )
        self.estimator = nn.Linear(hidden_layer_sizes[-1], 1)

    def self_attention(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.w_query(x)
        K = self.w_key(x)
        V = self.w_value(x)

        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(2, 1)) / (d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        x_att = torch.bmm(attn_weights, V)
        return x_att

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x_feat = self.extractor(x)
        x_att = self.self_attention(x_feat)

        logits = self.detector(x_feat)
        logits = logits.transpose(2, 1)

        masked_logits = logits.masked_fill(m.transpose(2, 1) == 0, float('-inf'))
        weights = torch.softmax(masked_logits, dim=2)

        bag_embedding = torch.bmm(weights, x_att)
        bag_score = self.estimator(bag_embedding)
        bag_pred = self.get_pred(bag_score)

        return weights, bag_pred

class HopfieldAttentionNetwork(BaseNetwork):
    def __init__(self, beta: float = 1.0, **kwargs):
        self.beta = beta
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.query_vector = nn.Parameter(torch.randn(1, hidden_layer_sizes[-1]))
        self.estimator = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        B, N, _ = x.size()

        x_feat = self.extractor(x)

        q = self.query_vector.unsqueeze(0).expand(B, 1, -1)

        logits = self.beta * torch.bmm(q, x_feat.transpose(2, 1))

        mask_bool = m.squeeze(-1).bool()
        masked_logits = logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))
        weights = torch.softmax(masked_logits, dim=2)

        bag_embedding = torch.bmm(weights, x_feat)
        bag_score = self.estimator(bag_embedding)
        bag_pred = self.get_pred(bag_score)

        return weights, bag_pred
