"""The configuration is fully wired, and stays that way.

Three properties, all checked reflectively over the dataclass tree so they cannot
go stale as fields move:

1. **Bijection** -- no documented key without a field, no field without a key.
2. **Every field has a consumer** -- a field nothing reads is documentation
   pretending to be configuration. An unread ``[cmodes]`` FedYogi block would
   leave the reported constant beta_2 = 0.999 matching the code by coincidence,
   and editing the file would not change the run.
3. **Validation rejects what the code cannot honour**, rather than failing later
   in a way that looks like a modelling problem.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

import koa_multimodal
from koa_multimodal.config.loader import field_paths, leaf_keys, load_config
from koa_multimodal.config.paths import default_config_path
from koa_multimodal.config.schema import (
    CmodesConfig,
    Config,
    ModelConfig,
    RckfConfig,
    StatisticsConfig,
    TrainingConfig,
)
from koa_multimodal.core.errors import ConfigError, KoaWarning

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PACKAGE_ROOT = Path(koa_multimodal.__file__).resolve().parent


def raw_toml():
    return tomllib.loads(default_config_path().read_text(encoding="utf-8"))


def test_toml_and_schema_are_in_bijection():
    documented = {".".join(path) for path in leaf_keys(raw_toml())}
    declared = {".".join(path) for path in field_paths()}
    assert documented == declared, (
        f"missing from TOML: {sorted(declared - documented)}; "
        f"extra in TOML: {sorted(documented - declared)}"
    )


def test_every_config_field_has_a_consumer():
    """A field nothing reads is documentation pretending to be configuration.

    The scan covers the whole package including ``config/`` itself, because a
    derived property (``InputConfig.mri_shape``) or a ``validate()`` guard is a
    legitimate consumer. It still catches genuinely dead fields: a bare dataclass
    declaration is ``name: type = default``, with no leading dot, so it never
    matches the attribute-access pattern.
    """

    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    unread = [
        ".".join(path)
        for path in field_paths()
        if not re.search(r"\.%s\b" % re.escape(path[-1]), sources)
    ]
    assert not unread, f"configuration fields nobody reads: {unread}"


def test_unknown_keys_are_rejected(tmp_path):
    """A typo must be an error, not a silently ignored line.

    A key with no field is a typo or a leftover, and both are invisible without
    this check: the loader would simply skip the line and the file would go on
    reading as though it governed the run.
    """

    target = tmp_path / "typo.toml"
    target.write_text("[training]\nqat_batch_size = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown configuration keys"):
        load_config(target)


def test_a_valid_subset_still_loads(tmp_path):
    """Overrides are partial by design: the TOML need not restate every default."""

    target = tmp_path / "partial.toml"
    target.write_text("[rckf]\nprior_variance = 0.2\n", encoding="utf-8")
    config = load_config(target)
    assert config.rckf.prior_variance == 0.2
    assert config.rckf.variance_ceiling == 2.0  # untouched default


def test_defaults_load_and_validate(config):
    assert config.project.name == "KOA_Multimodal_v7"
    assert config.training.environment == "koa_project"
    assert config.training.precision == "fp32"


class TestThesisConstants:
    """Values the dissertation reports. Changing one means correcting the text too."""

    def test_rckf(self, config):
        assert config.rckf.prior_variance == 0.15  # sigma_x^2
        assert config.rckf.variance_floor == 0.05  # R_floor
        assert config.rckf.variance_ceiling == 2.0  # R_ceil
        assert config.rckf.residual_weight in (0.1, 0.2, 0.5)  # alpha_res search space

    def test_cmodes_fedyogi(self, config):
        assert config.cmodes.selector_beta1 == 0.9
        assert config.cmodes.selector_beta2 == 0.999

    def test_qat(self, config):
        assert config.training.qat_temperature == 3.0
        assert config.training.qat_selector_weight == 0.1

    def test_statistics(self, config):
        assert config.statistics.bootstrap_iterations == 2000
        assert config.statistics.bootstrap_alpha == 0.05
        assert config.statistics.cluster_by_subject is True

    def test_training(self, config):
        assert config.training.epochs == 20
        assert config.training.optimizer == "AdamW"
        assert config.training.seed == 42
        assert config.training.checkpoint_metric == "val_qwk"

    def test_input_shapes(self, config):
        assert config.input.xray_size == 384
        assert config.input.mri_shape == (32, 384, 384)
        assert config.model.num_classes == 5
        assert config.model.feature_dim == 64

    def test_mri_backbones_are_randomly_initialised(self, config):
        """§4.1.4 states this explicitly; enabling Kinetics-400 makes it false."""

        assert config.model.mri_pretrained is False


class TestValidation:
    def test_fusion_batch_size_must_allow_negatives(self):
        with pytest.raises(ConfigError, match="InfoNCE"):
            TrainingConfig(fusion_batch_size=1).validate()

    def test_variance_bounds_must_be_ordered(self):
        with pytest.raises(ConfigError, match="variance_floor < variance_ceiling"):
            RckfConfig(variance_floor=2.0, variance_ceiling=0.5).validate()

    def test_ceiling_must_exceed_prior(self):
        """Otherwise a missing modality would look more trustworthy than a present one."""

        broken = dataclasses.replace(
            Config(), rckf=RckfConfig(prior_variance=3.0, variance_ceiling=2.0)
        )
        with pytest.raises(ConfigError, match="variance_ceiling must exceed"):
            broken.validate()

    def test_variance_groups_must_divide_feature_dim(self):
        broken = dataclasses.replace(
            Config(), rckf=RckfConfig(variance_groups=7), model=ModelConfig(feature_dim=64)
        )
        with pytest.raises(ConfigError, match="must divide"):
            broken.validate()

    def test_boundary_bins_must_be_sorted(self):
        with pytest.raises(ConfigError, match="sorted"):
            CmodesConfig(boundary_bins=(0.30, 0.10)).validate()

    def test_bootstrap_iterations_have_a_floor(self):
        with pytest.raises(ConfigError, match="at least 1000"):
            StatisticsConfig(bootstrap_iterations=100).validate()

    def test_gelu_warns_that_it_voids_monotonicity(self):
        with pytest.warns(KoaWarning, match="monotonicity"):
            RckfConfig(monotone_activation="gelu").validate()

    def test_warmup_must_be_shorter_than_training(self):
        with pytest.raises(ConfigError, match="warmup_epochs"):
            TrainingConfig(epochs=5, warmup_epochs=5).validate()
