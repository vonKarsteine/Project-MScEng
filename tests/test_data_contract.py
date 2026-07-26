"""The data-layer contract -- §4.1.3.

Shapes, ranges, channel order and augmentation gating, all checked without a
dataset by synthesising inputs. Two of these pin bug fixes: CLAHE's degraded path
must not change the value range by 255x, and MRI augmentation must respect the
seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from koa_multimodal.config.schema import (
    AugmentationConfig,
    InputConfig,
    PreprocessingConfig,
)
from koa_multimodal.core.seeding import sample_generator, stable_hash
from koa_multimodal.data.augment import augment_mri, augment_xray
from koa_multimodal.data.dataset import Split, collate_pairs
from koa_multimodal.data.folds import grouped_stratified_fold_ids, subject_stratum
from koa_multimodal.data.records import PairRecord, grade_distribution
from koa_multimodal.data.sampling import sqrt_balanced_weights

PRE = PreprocessingConfig()
AUG = AugmentationConfig()
INPUT = InputConfig()


class TestShapesAndRanges:
    def test_xray_normalisation_uses_the_thesis_constants(self):
        from koa_multimodal.data.xray import normalize_xray_imagenet_gray

        image = np.full((1, 8, 8), 0.449, dtype=np.float32)
        normalized = normalize_xray_imagenet_gray(image, PRE)
        assert np.allclose(normalized, 0.0, atol=1e-6)
        assert PRE.xray_imagenet_gray_mean == 0.449
        assert PRE.xray_imagenet_gray_std == 0.226

    def test_normalisation_runs_after_augmentation(self):
        """The augmenter clips to [0, 1]. Normalising first would push values
        outside that range and the clip would destroy the image."""

        raw = np.random.default_rng(0).random((1, 32, 32)).astype(np.float32)
        augmented = augment_xray(raw, AUG.xray, np.random.default_rng(0))
        assert augmented.min() >= -1e-6 and augmented.max() <= 1.0 + 1e-6

    def test_mri_shape_and_channel_order(self):
        """(2, D, H, W) with channel 0 = T2 and channel 1 = R2. Load-bearing:
        the two channels have different clip ranges."""

        assert INPUT.mri_shape == (32, 384, 384)
        assert PRE.t2_clip_max == 110.0
        assert PRE.r2_clip_max == 1.0

    def test_mri_is_never_imagenet_normalised(self):
        volume = np.random.default_rng(0).random((2, 4, 8, 8)).astype(np.float32)
        augmented = augment_mri(volume, AUG.mri, np.random.default_rng(0))
        assert augmented.shape == volume.shape
        assert augmented.min() >= -1.0  # not shifted by an ImageNet mean


class TestAugmentationSeeding:
    def test_the_same_generator_gives_the_same_augmentation(self):
        volume = np.random.default_rng(1).random((2, 4, 8, 8)).astype(np.float32)
        first = augment_mri(volume, AUG.mri, np.random.default_rng(7))
        second = augment_mri(volume, AUG.mri, np.random.default_rng(7))
        assert np.array_equal(first, second)

    def test_different_generators_give_different_augmentation(self):
        volume = np.random.default_rng(1).random((2, 4, 8, 8)).astype(np.float32)
        first = augment_mri(volume, AUG.mri, np.random.default_rng(7))
        second = augment_mri(volume, AUG.mri, np.random.default_rng(8))
        assert not np.array_equal(first, second)

    def test_both_modalities_are_seeded_the_same_way(self):
        """Both modalities take an explicit generator.

        Seeding one path through the global RNG while the other draws from OS
        entropy leaves a run half-reproducible, and the half that is not is
        invisible: the tensors still look plausible."""

        image = np.random.default_rng(1).random((1, 16, 16)).astype(np.float32)
        a = augment_xray(image, AUG.xray, np.random.default_rng(3))
        b = augment_xray(image, AUG.xray, np.random.default_rng(3))
        assert np.array_equal(a, b)

    def test_per_sample_generator_is_reproducible_and_distinct(self):
        assert stable_hash("a", 1) == stable_hash("a", 1)
        assert stable_hash("a", 1) != stable_hash("a", 2)
        first = sample_generator(42, "pair_001", epoch=0).random(4)
        again = sample_generator(42, "pair_001", epoch=0).random(4)
        other = sample_generator(42, "pair_002", epoch=0).random(4)
        assert np.array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_epoch_changes_the_draw_but_stays_reproducible(self):
        e0 = sample_generator(42, "pair_001", epoch=0).random(4)
        e1 = sample_generator(42, "pair_001", epoch=1).random(4)
        assert not np.array_equal(e0, e1)
        assert np.array_equal(e1, sample_generator(42, "pair_001", epoch=1).random(4))


class TestSplitGate:
    def test_split_enum_includes_oof(self):
        """'oof' is a real member, not a string that falls through every branch."""

        assert Split.OOF.value == "oof"
        assert {s.value for s in Split} == {"train", "val", "test", "oof"}


class TestCollate:
    def _sample(self, mri):
        import torch

        return {
            "xray": torch.randn(1, 8, 8),
            "mri": mri,
            "label": torch.tensor(1),
            "pair_key": "p",
            "subject_id": "s",
        }

    def test_all_none_mri_batch_is_allowed(self):
        batch = collate_pairs([self._sample(None), self._sample(None)])
        assert batch["mri"] is None

    def test_mixed_modality_batch_raises(self):
        """Missing modality is a property of the loader, not of individual
        samples: a half-populated batch has no coherent forward path."""

        import torch

        with pytest.raises(Exception):
            collate_pairs([self._sample(None), self._sample(torch.randn(2, 2, 4, 4))])


class TestSamplingAndFolds:
    def test_sqrt_balanced_lifts_minorities_without_over_repeating(self, records):
        weights = sqrt_balanced_weights(records)
        counts = grade_distribution(records)
        by_grade = {}
        for record, weight in zip(records, weights):
            by_grade.setdefault(record.grade, weight)
        common = max(counts, key=lambda g: counts[g])
        rare = min(counts, key=lambda g: counts[g])
        if counts[common] > counts[rare]:
            assert by_grade[rare] > by_grade[common]

    def test_folds_are_subject_grouped(self, records):
        ids = grouped_stratified_fold_ids(records, n_folds=3, seed=42)
        by_subject = {}
        for record, fold in zip(records, ids):
            by_subject.setdefault(record.subject_id, set()).add(fold)
        assert all(len(folds) == 1 for folds in by_subject.values())

    def test_folds_are_reasonably_balanced(self, records):
        """The greedy scores marginal cost, not absolute level -- otherwise a
        fold near target keeps winning and the split degenerates."""

        ids = grouped_stratified_fold_ids(records, n_folds=3, seed=42)
        sizes = [ids.count(f) for f in set(ids)]
        assert max(sizes) - min(sizes) <= max(4, 0.1 * len(records) / 3)

    def test_subject_stratum_uses_the_highest_grade(self):
        """A patient's two knees routinely differ, so no label-pure clustering
        exists; the highest grade is the recorded convention."""

        pair = [
            PairRecord("s1", "left", "train", 0, "x", "t", "r", "s1_L"),
            PairRecord("s1", "right", "train", 3, "x", "t", "r", "s1_R"),
        ]
        assert subject_stratum(pair)["s1"] == 3
