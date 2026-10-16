import numpy as np
import pytest
import torch

from pra import (
    ContentOnlyAggregation,
    LandmarkRegionSpec,
    PeriocularResponseAggregation,
    neutralize_periocular_motion,
)


def test_pra_masks_invalid_tokens_and_uses_original_tokens() -> None:
    torch.manual_seed(7)
    layer = PeriocularResponseAggregation(token_dim=4, hidden_dim=6)
    original = torch.randn(2, 5, 4)
    neutralized = original.clone()
    neutralized[:, 1, 0] += 1.0
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

    pooled, details = layer(original, neutralized, mask)

    assert pooled.shape == (2, 6)
    assert details["attention"].shape == (2, 5)
    assert torch.allclose(details["attention"][~mask], torch.zeros_like(details["attention"][~mask]))
    assert torch.allclose(details["attention"].sum(dim=1), torch.ones(2))
    assert details["response"][:, 1].gt(0).all()


def test_pra_adds_only_one_parameter_to_matched_aggregation() -> None:
    baseline = ContentOnlyAggregation(token_dim=4, hidden_dim=6)
    pra = PeriocularResponseAggregation(token_dim=4, hidden_dim=6)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    pra_count = sum(parameter.numel() for parameter in pra.parameters())
    assert pra_count == baseline_count + 1


def test_pra_rejects_samples_without_valid_tokens() -> None:
    layer = PeriocularResponseAggregation(token_dim=4, hidden_dim=6)
    tokens = torch.randn(1, 2, 4)
    with pytest.raises(ValueError, match="at least one valid token"):
        layer(tokens, tokens, torch.zeros(1, 2, dtype=torch.bool))


def test_periocular_neutralization_returns_image_and_soft_mask() -> None:
    onset = np.zeros((32, 32, 3), dtype=np.uint8)
    apex = np.full((32, 32, 3), 200, dtype=np.uint8)
    landmarks = np.array(
        [[8, 12], [12, 12], [20, 12], [24, 12], [16, 18]], dtype=np.float32
    )
    spec = LandmarkRegionSpec(
        left_eye=(0, 1),
        right_eye=(2, 3),
        regions=((0, 1, 4), (2, 3, 4)),
        anchors=(0, 1, 2, 3, 4),
    )

    neutralized, mask = neutralize_periocular_motion(
        onset,
        apex,
        landmarks,
        landmarks,
        spec,
        margin_ratio=0.0,
        feather_ratio=0.0,
    )

    assert neutralized.shape == apex.shape
    assert mask.shape == apex.shape[:2]
    assert mask.max() == 255
    assert neutralized[mask > 0].max() == 0
    assert neutralized[mask == 0].min() == 200

