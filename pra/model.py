from __future__ import annotations

import torch
from torch import nn

from .token_aggregation import ContentOnlyAggregation, PeriocularResponseAggregation


class ContentOnlyClassifier(nn.Module):
    """Content-only baseline with a linear classification head."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.aggregation = ContentOnlyAggregation(token_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        original_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled, details = self.aggregation(original_tokens, token_mask)
        return self.classifier(pooled), details


class PRAClassifier(nn.Module):
    """PRA aggregation followed by a linear classification head."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.aggregation = PeriocularResponseAggregation(token_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        original_tokens: torch.Tensor,
        neutralized_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        neutralized_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled, details = self.aggregation(
            original_tokens,
            neutralized_tokens,
            token_mask,
            neutralized_token_mask,
        )
        return self.classifier(pooled), details

