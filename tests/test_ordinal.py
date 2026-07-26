"""The CORN conditional-ordinal head -- eq. 3.1-3.2."""

from __future__ import annotations

import pytest
import torch

from koa_multimodal.core.errors import ContractError
from koa_multimodal.core.ordinal import (
    CornOrdinalHead,
    corn_logits_to_probabilities,
    corn_ordinal_loss,
    expected_grade,
    normalized_entropy,
    probability_nll_loss,
    top2_margin,
)

NUM_CLASSES = 5


def test_head_emits_k_minus_one_threshold_logits():
    """Four conditional thresholds for five grades -- not five class logits."""

    head = CornOrdinalHead(input_dim=16, num_classes=NUM_CLASSES)
    assert head(torch.randn(3, 16)).shape == (3, NUM_CLASSES - 1)


def test_chain_rule_produces_a_normalised_posterior():
    probabilities = corn_logits_to_probabilities(torch.randn(8, 4))
    assert probabilities.shape == (8, NUM_CLASSES)
    assert torch.allclose(probabilities.sum(1), torch.ones(8), atol=1e-5)
    assert (probabilities >= 0).all()


def test_chain_rule_matches_the_closed_form():
    """P(KL0)=1-q0, P(KL1)=q0(1-q1), ..., P(KL4)=q0q1q2q3."""

    logits = torch.tensor([[0.5, -0.25, 1.0, -1.5]])
    q = torch.sigmoid(logits)[0]
    expected = torch.tensor(
        [
            1 - q[0],
            q[0] * (1 - q[1]),
            q[0] * q[1] * (1 - q[2]),
            q[0] * q[1] * q[2] * (1 - q[3]),
            q[0] * q[1] * q[2] * q[3],
        ]
    )
    got = corn_logits_to_probabilities(logits)[0]
    assert torch.allclose(got, expected / expected.sum(), atol=1e-6)


def test_monotone_thresholds_give_a_monotone_severity_shift():
    """Raising every threshold logit must move mass toward higher grades."""

    low = corn_logits_to_probabilities(torch.full((1, 4), -2.0))
    high = corn_logits_to_probabilities(torch.full((1, 4), 2.0))
    assert expected_grade(high) > expected_grade(low)


def test_loss_supervises_only_reached_thresholds():
    """q_j is conditional on y > KL_{j-1}, so threshold j is supervised only on
    samples with y >= j. A label of 0 therefore constrains one threshold."""

    logits = torch.zeros(1, 4, requires_grad=True)
    corn_ordinal_loss(logits, torch.tensor([0])).backward()
    grad = logits.grad[0]
    assert grad[0].abs() > 0
    assert torch.allclose(grad[1:], torch.zeros(3))


def test_loss_is_finite_across_the_grade_range():
    logits = torch.randn(5, 4)
    for label in range(NUM_CLASSES):
        loss = corn_ordinal_loss(logits, torch.full((5,), label))
        assert torch.isfinite(loss)


def test_loss_rejects_a_wrong_rank():
    with pytest.raises(ContractError):
        corn_ordinal_loss(torch.randn(4, 5, 2), torch.tensor([0, 1, 2, 3]))


def test_normalized_entropy_spans_zero_to_one():
    certain = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    uniform = torch.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES)
    assert float(normalized_entropy(certain)) == pytest.approx(0.0, abs=1e-4)
    assert float(normalized_entropy(uniform)) == pytest.approx(1.0, abs=1e-5)


def test_expected_grade_and_margin_match_their_definitions():
    probabilities = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.3, 0.2, 0.0, 0.0]])
    assert torch.allclose(expected_grade(probabilities), torch.tensor([2.0, 0.7]), atol=1e-6)
    assert torch.allclose(top2_margin(probabilities), torch.tensor([1.0, 0.2]), atol=1e-6)


def test_probability_nll_is_the_posterior_route_objective():
    probabilities = torch.tensor([[0.9, 0.05, 0.03, 0.01, 0.01]])
    confident = probability_nll_loss(probabilities, torch.tensor([0]))
    wrong = probability_nll_loss(probabilities, torch.tensor([4]))
    assert wrong > confident


def test_head_rejects_too_few_classes():
    with pytest.raises(ContractError):
        CornOrdinalHead(input_dim=8, num_classes=2)
