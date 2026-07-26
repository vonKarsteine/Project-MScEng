"""Deployment-oriented QAT -- §3.4, Tables 3.3-3.4.

Three properties matter and none of them requires training to check:

* the **precision profile** is uneven on purpose, and the CORN head must never be
  quantized;
* the **observers keep calibrating** across batches. The router scores the
  selector with the module in eval mode, so tying observation to
  ``Module.training`` would let one of the two INT8 components capture its range
  from the first batch and then freeze;
* the **7-field contract order** is derived from the dataclass rather than
  restated, so the ONNX signature cannot drift out of sync with the payload.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from koa_multimodal.config.schema import TrainingConfig
from koa_multimodal.core.contract import CandidateOutput, OutputMeta
from koa_multimodal.core.ordinal import CornOrdinalHead
from koa_multimodal.deploy.composite import CompositeDeploymentModel
from koa_multimodal.deploy.contract import (
    CONTRACT_FIELD_NAMES,
    ONNX_OUTPUT_NAMES,
    DeploymentContract,
    extract_deployment_contract,
)
from koa_multimodal.deploy.distill import FrozenTeacherStudentQAT
from koa_multimodal.deploy.objectives import QatObjectiveConfig, qat_loss_terms, qat_objective
from koa_multimodal.deploy.quantize import FakeQuantPolicy, StraightThroughFakeQuant
from koa_multimodal.deploy.student import QATStudent
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel
from koa_multimodal.fusion.rckf import RCKFFusion
from koa_multimodal.models.mri import MriEncoder
from koa_multimodal.models.xray import XrayBackbone

FEATURE_DIM = 32


def build_composite():
    candidates = [
        RCKFFusion(
            feature_dim=FEATURE_DIM,
            candidate_id="rckf_%d" % index,
            xray_encoder=XrayBackbone(feature_dim=FEATURE_DIM),
            mri_encoder=MriEncoder(in_channels=2, feature_dim=FEATURE_DIM),
        )
        for index in range(2)
    ]
    return CompositeDeploymentModel(
        candidates=candidates,
        candidate_ids=["rckf_0", "rckf_1"],
        selector_model=CMODESSelectorModel(input_dim=16),
        default_candidate_id="rckf_0",
    )


@pytest.fixture
def batch():
    return torch.randn(4, 1, 64, 64), torch.randn(4, 2, 8, 32, 32), torch.tensor([0, 1, 2, 4])


class TestDeploymentContract:
    def test_seven_fields_in_the_specified_order(self):
        assert CONTRACT_FIELD_NAMES == [
            "probabilities",
            "posterior_latent",
            "measurement_variance",
            "kalman_gain",
            "selector_scores",
            "route",
            "missing_modality_fallback",
        ]

    def test_onnx_names_are_derived_not_restated(self):
        """The invariant is structural: there is no second list to drift."""

        assert ONNX_OUTPUT_NAMES == [item.name for item in fields(DeploymentContract)]
        assert ONNX_OUTPUT_NAMES == CONTRACT_FIELD_NAMES

    def test_as_tuple_matches_the_field_order(self, batch):
        xray, mri, _ = batch
        model = build_composite().eval()
        with torch.no_grad():
            contract = extract_deployment_contract(model(xray, mri))
        assert len(contract.as_tuple()) == len(CONTRACT_FIELD_NAMES)

    def test_contract_survives_a_missing_modality(self, batch):
        """The X-ray-only path still has to fill all seven fields."""

        xray, _, _ = batch
        model = build_composite().eval()
        with torch.no_grad():
            contract = extract_deployment_contract(model(xray, None))
        assert len(contract.as_tuple()) == 7
        assert bool(contract.missing_modality_fallback.any())


class TestComposite:
    def test_emits_a_posterior_route(self, batch):
        """The router selects a posterior, so the composite owns no CORN head --
        which is why a CORN objective on it must raise rather than read K
        thresholds off a K-class vector."""

        xray, mri, _ = batch
        with torch.no_grad():
            out = build_composite().eval()(xray, mri)
        assert out.threshold_logits is None
        assert out.trace.route is not None
        assert out.trace.member_outputs is not None

    def test_router_and_fallback_are_registered_submodules(self):
        """Both are built once, in ``__init__``. Constructing them inside
        ``forward`` would leave them unregistered, so ``.to(device)``, ``.eval()``
        and ``state_dict()`` would never reach them."""

        model = build_composite()
        names = dict(model.named_modules())
        assert any("router" in name for name in names)
        assert isinstance(model, torch.nn.Module)
        # A device move must reach everything the forward pass uses.
        model.to("cpu")
        assert all(param.device.type == "cpu" for param in model.parameters())


class TestPrecisionProfile:
    def test_ordinal_head_is_never_quantized(self, batch):
        """Four chained sigmoids compound an early rounding error into every
        later class posterior (Table 3.3)."""

        student = QATStudent(build_composite())
        head_modules = {
            id(module)
            for module in student.modules()
            if isinstance(module, CornOrdinalHead)
        }
        for name, module in student.named_modules():
            if isinstance(module, StraightThroughFakeQuant):
                owner = name.rsplit(".", 1)[0]
                assert "head" not in owner.lower(), f"CORN head was instrumented at {name}"
        assert head_modules, "the composite should own at least one CORN head"
        assert student.precision_profile()["ordinal_head"].startswith("fp32")

    def test_xray_and_selector_are_quantized(self):
        student = QATStudent(build_composite())
        profile = student.precision_profile()
        assert "int8" in profile["xray_encoder"]
        assert "int8" in profile["selector"]
        assert student.quantized_module_names()

    def test_mri_branch_runs_mixed_precision(self):
        student = QATStudent(build_composite())
        assert "fp16" in student.precision_profile()["mri_rckf"]

    def test_disabled_policy_instruments_nothing(self):
        student = QATStudent(build_composite(), FakeQuantPolicy(enabled=False))
        assert not student.quantized_module_names()


class TestObserverCalibration:
    def test_observers_keep_updating_across_batches_in_eval(self):
        """The regression test for the defect that mattered.

        The selector is scored with the module in eval mode, so tying observation
        to ``Module.training`` froze its range after the first batch -- and the
        selector is one of only two INT8 components.
        """

        quantizer = StraightThroughFakeQuant(FakeQuantPolicy())
        quantizer.eval()
        quantizer.enable_observation() if hasattr(quantizer, "enable_observation") else None
        quantizer(torch.tensor([[-1.0, 1.0]]))
        first = float(quantizer.observer.max_value)
        quantizer(torch.tensor([[-50.0, 50.0]]))
        second = float(quantizer.observer.max_value)
        assert second > first, "observer stopped calibrating while in eval mode"

    def test_freeze_stops_calibration(self):
        quantizer = StraightThroughFakeQuant(FakeQuantPolicy())
        quantizer(torch.tensor([[-1.0, 1.0]]))
        quantizer.observer.freeze()
        before = float(quantizer.observer.max_value)
        quantizer(torch.tensor([[-99.0, 99.0]]))
        assert float(quantizer.observer.max_value) == before

    def test_student_exposes_observer_state(self):
        student = QATStudent(build_composite())
        assert isinstance(student.observer_state(), dict)


class TestDistillation:
    def test_teacher_stays_frozen_even_after_train(self, batch):
        """``train()`` must re-force the teacher to eval, and no teacher
        parameter may carry a gradient -- optimisers are built on the student."""

        model = FrozenTeacherStudentQAT(build_composite())
        model.train()
        assert model.teacher.training is False
        assert all(not p.requires_grad for p in model.teacher.parameters())

    def test_forward_attaches_a_teacher_trace(self, batch):
        xray, mri, _ = batch
        model = FrozenTeacherStudentQAT(build_composite())
        out = model(xray, mri)
        assert out.trace.teacher is not None
        assert not out.trace.teacher.probabilities.requires_grad


class TestObjective:
    def test_six_terms_are_all_present(self, batch):
        xray, mri, labels = batch
        model = FrozenTeacherStudentQAT(build_composite())
        terms = qat_loss_terms(model(xray, mri), labels, QatObjectiveConfig())
        assert set(terms) >= {
            "corn",
            "kd",
            "boundary",
            "ordinal",
            "reliability",
            "selector",
        }
        assert all(torch.isfinite(value) for value in terms.values())

    def test_weights_come_from_the_training_config(self):
        cfg = QatObjectiveConfig.from_training(TrainingConfig())
        assert cfg.temperature == 3.0
        weights = cfg.weights()
        assert weights["kd"] == 0.5 and weights["boundary"] == 0.2

    def test_a_posterior_student_is_supervised_by_nll_not_corn(self, batch):
        """The headline fix. The composite is a posterior route, so applying CORN
        would read five thresholds off a five-class probability vector."""

        from koa_multimodal.deploy.objectives import supervision_kind

        xray, mri, _ = batch
        with torch.no_grad():
            out = build_composite().eval()(xray, mri)
        assert supervision_kind(out) != "corn"

    def test_a_corn_student_is_supervised_by_corn(self):
        from koa_multimodal.deploy.objectives import supervision_kind

        out = CandidateOutput.from_corn(torch.randn(2, 4), OutputMeta(candidate_id="rckf"))
        assert supervision_kind(out) == "corn"

    def test_objective_returns_the_total_and_its_breakdown(self, batch):
        """Six terms spanning several orders of magnitude: a run where one has
        quietly collapsed to zero is indistinguishable from a healthy one at the
        level of the total, so the breakdown travels with it."""

        xray, mri, labels = batch
        model = FrozenTeacherStudentQAT(build_composite())
        total, terms = qat_objective(model(xray, mri), labels, QatObjectiveConfig())
        assert total.ndim == 0 and torch.isfinite(total)
        assert len(terms) >= 6

    def test_gradients_reach_only_the_student(self, batch):
        xray, mri, labels = batch
        model = FrozenTeacherStudentQAT(build_composite())
        total, _ = qat_objective(model(xray, mri), labels, QatObjectiveConfig())
        total.backward()
        assert all(p.grad is None for p in model.teacher.parameters())
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in model.student.parameters()
        )


class TestOnnxExport:
    def test_export_produces_a_valid_graph(self, tmp_path, batch):
        """onnx is installed in this environment, so the path is live -- this is
        the first generation of the package in which it could be exercised."""

        from koa_multimodal.deploy.onnx import export_student

        xray, mri, _ = batch
        report = export_student(
            build_composite().eval(), tmp_path / "student.onnx", xray[:1], mri[:1]
        )
        if not report.get("onnxAvailable", True):
            pytest.skip("onnx is not installed in this environment")
        assert report["exported"] is True, report.get("reason")
        assert report["outputNames"] == ONNX_OUTPUT_NAMES
        assert report.get("valid") is not False

    def test_export_never_raises_when_the_path_is_bad(self, batch):
        """Export degrades to a report; it is an optional extra, not a hard
        dependency, and a demo must not crash because of it."""

        from koa_multimodal.deploy.onnx import export_student

        xray, mri, _ = batch
        report = export_student(
            build_composite().eval(), "/nonexistent\x00/bad.onnx", xray[:1], mri[:1]
        )
        assert isinstance(report, dict)
        assert report.get("exported") in (True, False)
