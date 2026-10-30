from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neonatal_sleep.features import extract_86_features
from neonatal_sleep.preprocessing import extract_epoch_windows, realign_candidate_epochs


def test_feature_contract() -> None:
    values = np.abs(np.random.default_rng(4).normal(size=(5, 2, 30))).astype(np.float32)
    features, names = extract_86_features(values)
    assert features.shape == (5, 86)
    assert len(names) == 86
    assert np.isfinite(features).all()


def test_label_realignment_selects_ordered_subsequence() -> None:
    candidate_labels = np.asarray([0, 1, 1, 2, 3, 0], dtype=np.int32)
    candidate_starts = np.arange(6, dtype=float) * 30.0
    expected = np.asarray([1, 2, 3], dtype=np.int32)
    labels, starts = realign_candidate_epochs(candidate_labels, candidate_starts, expected)
    assert np.array_equal(labels, expected)
    assert np.array_equal(starts, np.asarray([30.0, 90.0, 120.0]))


def test_edf_windows_use_realigned_starts() -> None:
    signal = np.arange(5 * 3000, dtype=float)
    windows = extract_epoch_windows([signal], np.asarray([30.0, 90.0]))
    assert windows.shape == (2, 1, 3000)
    assert windows[0, 0, 0] == 3000
    assert windows[1, 0, 0] == 9000
