from __future__ import annotations

import torch
from torch import nn


def _validate_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (batch, token, channel), got {tokens.shape}")
    if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape (batch, token)")
    if not mask.any(dim=1).all():
        raise ValueError("each sample must contain at least one valid token")


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply softmax over valid token positions only."""

    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return scores.softmax(dim=1)


class _SharedTokenProjection(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.content_score = nn.Linear(hidden_dim, 1)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens)

    def pool(
        self,
        encoded: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = masked_softmax(scores, mask)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        return pooled, attention


class ContentOnlyAggregation(_SharedTokenProjection):
    """Matched baseline using original-token content scores only."""

    def __init__(self, token_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__(token_dim, hidden_dim, dropout)

    def forward(
        self,
        original_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_tokens(original_tokens, token_mask)
        encoded = self.encode(original_tokens)
        content_scores = self.content_score(encoded).squeeze(-1)
        pooled, attention = self.pool(encoded, content_scores, token_mask)
        return pooled, {"attention": attention, "content_scores": content_scores}


class PeriocularResponseAggregation(_SharedTokenProjection):
    """Response-guided aggregation using only original tokens in the final pool."""

    def __init__(self, token_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__(token_dim, hidden_dim, dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        original_tokens: torch.Tensor,
        neutralized_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        neutralized_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_tokens(original_tokens, token_mask)
        if neutralized_tokens.shape != original_tokens.shape:
            raise ValueError("original and neutralized tokens must have matching shapes")
        if neutralized_token_mask is None:
            neutralized_token_mask = token_mask
        _validate_tokens(neutralized_tokens, neutralized_token_mask)

        valid_mask = token_mask & neutralized_token_mask
        if not valid_mask.any(dim=1).all():
            raise ValueError("each sample must share at least one valid token")

        original_encoded = self.encode(original_tokens)
        neutralized_encoded = self.encode(neutralized_tokens)
        response = torch.linalg.vector_norm(
            original_encoded - neutralized_encoded, dim=-1
        )
        content_scores = self.content_score(original_encoded).squeeze(-1)
        scores = content_scores + self.alpha * response
        pooled, attention = self.pool(original_encoded, scores, valid_mask)

        response_mean = (response * valid_mask.to(response.dtype)).sum(dim=1)
        response_mean /= valid_mask.sum(dim=1).clamp_min(1)
        return pooled, {
            "attention": attention,
            "content_scores": content_scores,
            "response": response,
            "response_mean": response_mean,
        }

