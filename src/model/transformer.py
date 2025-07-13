import torch
from torch import nn
from torch import Tensor
from typing import Optional

from .utils import ActivationType, init_add_activation, point_rope

class TransformerEncoderLayer(nn.Module):
    @init_add_activation
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # input shape: (batch_size, n_points, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        position: Tensor,  # shape: (batch_size, n_points, 3)
        tokens: Tensor,  # shape: (batch_size, n_points + 1, d_model)
        src_mask: Optional[Tensor] = None,  # shape: (n_points, n_points)
        src_key_padding_mask: Optional[Tensor] = None  # shape: (batch_size, n_points)
    ) -> Tensor:
        input_tokens = tokens

        tokens_with_pos = torch.cat([point_rope(tokens[:, :-1, :], position), tokens[:, -1, :].unsqueeze(1)], dim=1)
        attn_output, _ = self.self_attn(tokens_with_pos, tokens_with_pos, tokens_with_pos,
                                    attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        tokens = input_tokens + self.dropout1(attn_output)
        tokens = self.norm1(tokens)
        ff_output = self.linear2(self.dropout(self.activation_fn(self.linear1(tokens))))

        tokens = tokens + self.dropout2(ff_output)
        tokens = self.norm2(tokens)

        return tokens  # shape: (batch_size, n_points, d_model)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.RELU
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        position: Tensor,  # shape: (batch_size, n_points, 3)
        points: Tensor,  # shape: (batch_size, n_points, d_model)
        src_mask: Optional[Tensor] = None,  # shape: (n_points, n_points)
        src_key_padding_mask: Optional[Tensor] = None  # shape: (batch_size, n_points)
    ) -> Tensor:
        for layer in self.layers:
            points = layer(position, points, src_mask, src_key_padding_mask)
        return points  # shape: (batch_size, n_points, d_model)

class BinocularformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.RELU
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(1, d_model, padding_idx=0)
        # shape: (vocab_size=4, d_model)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

    def forward(
        self,
        position: Tensor,  # shape: (batch_size, n_points, 3)
        points: Tensor,  # shape: (batch_size, n_points, d_model)
        src_mask: Optional[Tensor] = None,  # shape: (n_points, n_points)
        src_key_padding_mask: Optional[Tensor] = None  # shape: (batch_size, n_points)
    ) -> Tensor:
        bs = points.shape[0]
        device = points.device

        cls_token = self.embedding(torch.full((bs, 1), 0, device=device))
        points = torch.cat([points, cls_token], dim=1)

        # shape: (batch_size, n_points + 3, d_model)
        return self.encoder(position, points, src_mask, src_key_padding_mask)
