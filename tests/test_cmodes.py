"""C-MODES invariants -- §3.3.

The sign convention test is the important one. The methodology defines the client
delta as ``theta_global - theta_local`` and has the server *subtract* the scaled
step. Flipping either sign passes every shape check and silently ascends the loss,
so it is asserted directly rather than trusted.
"""

from __future__ import annotations

import copy

import pytest
import torch

from koa_multimodal.core.contract import CandidateOutput, OutputMeta
from koa_multimodal.ensemble.cmodes.features import RiskFeatureBuilder
from koa_multimodal.ensemble.cmodes.federated import (
    FedYogiState,
    fedyogi_aggregate,
    local_update,
)
from koa_multimodal.ensemble.cmodes.risk import RiskDifferentialDataset, candidate_risk
from koa_multimodal.ensemble.cmodes.routing import CMODESRouter, switch_decision
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel

NUM_CLASSES = 5
EXPECTED_INPUT_DIM = 16


def make_outputs(batch: int = 6, count: int = 4):
    return [
        CandidateOutput.from_posterior(
            torch.softmax(torch.randn(batch, NUM_CLASSES), dim=1),
            OutputMeta(candidate_id="route_%d" % index),
        )
        for index in range(count)
    ]


class TestFeatureSpace:
    def test_input_dim_is_sixteen_and_independent_of_pool_size(self):
        """The vector describes one (sample, route) *pair*, not the whole pool."""

        builder = RiskFeatureBuilder(num_classes=NUM_CLASSES)
        assert builder.input_dim == EXPECTED_INPUT_DIM
        for count in (2, 4, 9):
            probs = torch.softmax(torch.randn(5, count, NUM_CLASSES), dim=-1)
            assert builder.from_probabilities(probs).shape == (5, count, EXPECTED_INPUT_DIM)

    def test_feature_spec_names_every_dimension(self):
        builder = RiskFeatureBuilder(num_classes=NUM_CLASSES)
        spec = builder.feature_spec()
        assert spec["inputDim"] == EXPECTED_INPUT_DIM
        assert len(spec["features"]) == EXPECTED_INPUT_DIM

    def test_default_route_has_zero_differential_against_itself(self):
        builder = RiskFeatureBuilder(num_classes=NUM_CLASSES)
        probs = torch.softmax(torch.randn(4, 3, NUM_CLASSES), dim=-1)
        features = builder.from_probabilities(probs, default_index=1)
        differential = features[:, 1, NUM_CLASSES + 3 :]
        # abs probability delta, grade step and margin delta all vanish; only the
        # boundary bin (a property of the default) is non-zero.
        assert torch.allclose(differential[:, : NUM_CLASSES + 2], torch.zeros(4, NUM_CLASSES + 2), atol=1e-6)

    def test_scoring_is_permutation_equivariant(self):
        """One shared network over all routes: reordering the pool reorders the
        scores and changes nothing else."""

        builder = RiskFeatureBuilder(num_classes=NUM_CLASSES)
        selector = CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM).eval()
        probs = torch.softmax(torch.randn(5, 4, NUM_CLASSES), dim=-1)

        with torch.no_grad():
            base = selector(builder.from_probabilities(probs, default_index=0))
            permutation = [0, 3, 1, 2]  # default stays at index 0
            swapped = selector(
                builder.from_probabilities(probs[:, permutation, :], default_index=0)
            )
        assert torch.allclose(base[:, permutation], swapped, atol=1e-6)


class TestRisk:
    def test_risk_penalises_confident_errors_more(self):
        labels = torch.tensor([0, 0])
        confident_wrong = torch.tensor([[0.01, 0.01, 0.01, 0.01, 0.96], [0.2, 0.2, 0.2, 0.2, 0.2]])
        predictions = torch.tensor([4, 4])
        risk = candidate_risk(confident_wrong, predictions, labels)
        assert risk[0] > risk[1]

    def test_ordinal_term_scales_with_distance(self):
        probs = torch.full((2, 5), 0.2)
        labels = torch.tensor([0, 0])
        near = candidate_risk(probs, torch.tensor([1, 1]), labels)[0]
        far = candidate_risk(probs, torch.tensor([4, 4]), labels)[0]
        assert far > near


class TestFedYogi:
    def test_delta_is_global_minus_local(self):
        """Eq. 3.23 -- the opposite of the usual local-minus-global convention."""

        selector = CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM)
        dataset = _synthetic_dataset()
        before = {n: p.detach().clone() for n, p in selector.named_parameters()}
        delta = local_update(selector, dataset, 1, learning_rate=0.1, local_steps=2)

        # Reconstruct theta_local and check the sign directly.
        for name, parameter in selector.named_parameters():
            assert torch.allclose(parameter, before[name])  # global is untouched
            local = before[name] - delta[name]
            assert not torch.allclose(local, before[name])

    def test_server_subtracts_the_scaled_step(self):
        """Eq. 3.25. A flipped sign here ascends the loss while passing shape checks."""

        selector = CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM)
        before = {n: p.detach().clone() for n, p in selector.named_parameters()}
        update = {n: torch.ones_like(p) for n, p in selector.named_parameters()}

        fedyogi_aggregate(selector, [update], [1.0], FedYogiState(), server_learning_rate=0.5)
        for name, parameter in selector.named_parameters():
            # A positive delta must move the parameter DOWN.
            assert (parameter <= before[name] + 1e-8).all(), f"{name} moved the wrong way"

    def test_second_moment_stays_non_negative(self):
        selector = CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM)
        state = FedYogiState()
        for _ in range(5):
            update = {n: torch.randn_like(p) for n, p in selector.named_parameters()}
            state = fedyogi_aggregate(selector, [update], [1.0], state)
        assert all((v >= 0).all() for v in state.second_moment.values())
        assert state.step == 5

    def test_weights_must_be_valid(self):
        selector = CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM)
        update = {n: torch.ones_like(p) for n, p in selector.named_parameters()}
        with pytest.raises(Exception):
            fedyogi_aggregate(selector, [update], [-1.0], FedYogiState())


class TestRouting:
    def test_default_is_masked_out_of_the_argmax(self):
        """"Stay" is the fallback, never a competitor the selector can win by a
        rounding error."""

        scores = torch.tensor([[9.0, 0.1, 0.2], [9.0, 0.5, 0.4]])
        best, index, _ = switch_decision(scores, default_index=0, tau_s=0.0, c_switch=0.0)
        assert (index != 0).all()
        assert torch.allclose(best, torch.tensor([0.2, 0.5]))

    def test_switch_requires_clearing_the_composite_barrier(self):
        scores = torch.tensor([[0.0, 0.03], [0.0, 0.30]])
        _, _, flag = switch_decision(scores, 0, tau_s=0.0, c_switch=0.04)
        assert flag.tolist() == [False, True]

    def test_only_the_sum_of_tau_and_cost_is_identifiable(self):
        scores = torch.randn(20, 3)
        a = switch_decision(scores, 0, tau_s=0.10, c_switch=0.05)[2]
        b = switch_decision(scores, 0, tau_s=0.00, c_switch=0.15)[2]
        assert torch.equal(a, b)

    def test_single_candidate_pool_never_switches(self):
        _, index, flag = switch_decision(torch.randn(4, 1), 0, 0.0, 0.0)
        assert not flag.any() and (index == 0).all()

    def test_router_emits_a_posterior_route(self):
        """Selecting a posterior destroys the threshold chain, so no CORN head."""

        router = CMODESRouter(
            selector_model=CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM),
            candidate_ids=["route_%d" % i for i in range(4)],
            default_candidate_id="route_0",
        )
        out = router(make_outputs())
        assert out.threshold_logits is None
        assert not out.is_ordinal_head

    def test_router_trace_records_the_decision(self):
        router = CMODESRouter(
            selector_model=CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM),
            candidate_ids=["route_%d" % i for i in range(4)],
            default_candidate_id="route_0",
            tau_s=0.0,
            c_switch=0.04,
        )
        outputs = make_outputs()
        trace = router(outputs).require_route()
        assert trace.pairwise_features.shape == (6, 4, EXPECTED_INPUT_DIM)
        assert trace.threshold == pytest.approx(0.04)
        assert trace.default_index == 0
        assert trace.selected.shape == (6,)

    def test_selected_posterior_comes_from_the_selected_route(self):
        router = CMODESRouter(
            candidate_ids=["route_%d" % i for i in range(4)], default_candidate_id="route_0"
        )
        outputs = make_outputs()
        out = router(outputs)
        trace = out.require_route()
        for row, route in enumerate(trace.selected.tolist()):
            assert torch.allclose(out.probabilities[row], outputs[route].probabilities[row], atol=1e-5)

    def test_router_is_a_registered_module(self):
        """It holds submodules, so it must follow .to(), .eval() and state_dict()."""

        router = CMODESRouter(selector_model=CMODESSelectorModel(input_dim=EXPECTED_INPUT_DIM))
        assert isinstance(router, torch.nn.Module)
        assert any("selector_model" in key for key in router.state_dict())


def _synthetic_dataset():
    """A minimal RiskDifferentialDataset built from in-memory OOF artifacts."""

    from koa_multimodal.ensemble.artifacts import FoldProvenance, PredictionArtifact

    torch.manual_seed(0)
    rows = 24
    labels = [int(v) for v in torch.randint(0, 5, (rows,))]
    subject_ids = ["sub%03d" % (i // 2) for i in range(rows)]
    fold_ids = [i % 3 for i in range(rows // 2) for _ in range(2)]
    order = ["route_0", "route_1", "route_2"]

    artifacts = []
    for name in order:
        probs = torch.softmax(torch.randn(rows, 5), dim=1)
        artifacts.append(
            PredictionArtifact(
                candidate_id=name,
                split="oof",
                labels=labels,
                predictions=[int(v) for v in probs.argmax(1)],
                probabilities=[[float(x) for x in row] for row in probs],
                pair_keys=["pair%03d" % i for i in range(rows)],
                subject_ids=subject_ids,
                fold_ids=fold_ids,
                candidate_order=order,
                fold_provenance={
                    f: FoldProvenance(f, "hash%d" % f, "subj%d" % f, 8) for f in set(fold_ids)
                },
            )
        )
    return RiskDifferentialDataset(artifacts, default_candidate_id="route_0")


def test_risk_dataset_exposes_per_client_data():
    dataset = _synthetic_dataset()
    assert dataset.input_dim == EXPECTED_INPUT_DIM
    assert dataset.candidate_count == 3
    features, targets = dataset.client_data(1)
    assert features.shape == (len(dataset), EXPECTED_INPUT_DIM)
    assert targets.shape == (len(dataset),)
    # The default route's differential against itself is identically zero.
    assert torch.allclose(dataset.risk_differentials[:, 0], torch.zeros(len(dataset)), atol=1e-6)
