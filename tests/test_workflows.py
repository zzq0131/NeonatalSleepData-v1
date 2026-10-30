from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neonatal_sleep.config import load_config
from neonatal_sleep.preprocessing import TARGET_CHANNELS, compute_epoch_aeeg


def run_cli(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *args], cwd=cwd,
        env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(cwd / "mpl")},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("script", ["prepare_data.py", "validate_data.py", "reproduce_statistics.py", "run_baselines.py"])
def test_entrypoints_work_outside_repository(script, tmp_path):
    assert "--data-root" in run_cli(script, "--help", cwd=tmp_path).stdout


def test_external_data_root_preserves_output_and_split_locations(tmp_path):
    dataset = tmp_path / "external"
    (dataset / "data").mkdir(parents=True)
    config = load_config(ROOT / "config.json", data_root=dataset)
    assert Path(config["data"]["eeg100_dir"]) == dataset / "data/eeg_100hz"
    assert Path(config["_project_root"]) == ROOT
    assert config["splits_file"] == "splits.json"
    assert config["outputs"]["prepared_root"] == "outputs/prepared"
    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "config.json", data_root=tmp_path / "missing")


def test_frozen_splits_separate_all_200_participants():
    splits = json.loads((ROOT / "splits.json").read_text())
    all_subjects = {f"NSD{i:04d}" for i in range(1, 201)}
    outer_sets = [set(fold["test_subjects"]) for fold in splits["outer_folds"]]
    assert len(outer_sets) == 5
    assert all(len(subjects) == 40 for subjects in outer_sets)
    assert set.union(*outer_sets) == all_subjects
    for i, outer in enumerate(splits["outer_folds"]):
        assert all(not outer_sets[i] & other for j, other in enumerate(outer_sets) if i != j)
        inner = splits["inner_folds"][str(outer["fold"]) ]
        inner_sets = [set(fold["test_subjects"]) for fold in inner]
        assert len(inner_sets) == 4
        assert all(not subjects & outer_sets[i] for subjects in inner_sets)
        assert set.union(*inner_sets) == all_subjects - outer_sets[i]
        assert sum(map(len, inner_sets)) == 160


def test_synthetic_preparation_validation_statistics_and_rf(tmp_path):
    # Four synthetic participants; no real clinical data are read or stored.
    dataset = tmp_path / "external"
    eeg_dir = dataset / "data/eeg_100hz"
    aeeg_dir = dataset / "data/aeeg_epoch"
    eeg_dir.mkdir(parents=True)
    aeeg_dir.mkdir()
    for index in range(1, 5):
        subject = f"NSD{index:04d}"
        rng = np.random.default_rng(index)
        x = rng.normal(0, 20e-6, size=(8, 8, 3000))
        y = np.tile(np.arange(4, dtype=np.int32), 2)
        aeeg = compute_epoch_aeeg(x, TARGET_CHANNELS)
        np.savez_compressed(eeg_dir / f"{subject}.npz", x=x, y=y,
                            sampling_rate=np.array(100), channel_names=np.array(TARGET_CHANNELS))
        np.savez_compressed(aeeg_dir / f"{subject}.npz", aeeg=aeeg, y=y,
                            source_sampling_rate_hz=np.array(100), aeeg_sampling_rate_hz=np.array(1),
                            source_window_seconds=np.array(30), bipolar_channel_names=np.array(["C3-P3", "C4-P4"]))
    config = json.loads((ROOT / "config.json").read_text())
    config["release"]["expected_subjects"] = 4
    config["baselines"]["rf"]["parameter_grid"] = [{"n_estimators": 5, "min_samples_leaf": 1}]
    config["baselines"]["bootstrap_repetitions"] = 5
    # A tiny two-fold split verifies evaluation plumbing, not paper performance.
    (tmp_path / "splits.json").write_text(json.dumps({"outer_folds": [
        {"fold": 0, "test_subjects": ["NSD0001", "NSD0002"]},
        {"fold": 1, "test_subjects": ["NSD0003", "NSD0004"]}], "inner_folds": {}}))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = ("--config", str(config_path), "--data-root", str(dataset))
    run_cli("prepare_data.py", *args, "--subject", "NSD0001", cwd=tmp_path)
    with np.load(tmp_path / "outputs/prepared/aeeg_epoch/NSD0001.npz") as prepared:
        with np.load(aeeg_dir / "NSD0001.npz") as expected:
            np.testing.assert_array_equal(prepared["y"], expected["y"])
            np.testing.assert_allclose(prepared["aeeg"], expected["aeeg"], rtol=0, atol=0)
    run_cli("validate_data.py", *args, cwd=tmp_path)
    validation = json.loads((tmp_path / "outputs/validation/validation_summary.json").read_text())
    assert validation["labelled_epoch_count"] == 32
    assert validation["stage_epoch_counts"] == {"W": 8, "N": 8, "R": 8, "T": 8}
    run_cli("reproduce_statistics.py", *args, cwd=tmp_path)
    assert (tmp_path / "outputs/statistics/figures/label_distribution.png").is_file()
    assert (tmp_path / "outputs/statistics/figures/participant_stage_features.png").is_file()
    import pandas as pd
    features = pd.read_csv(tmp_path / "outputs/statistics/epoch_features_86d.csv")
    assert features.shape == (32, 90)  # 86 features plus four index/label columns
    result = run_cli("run_baselines.py", *args, "--model", "random_forest", cwd=tmp_path)
    assert "random forest macro-F1" in result.stdout


def test_missing_edf_is_an_error(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "data").mkdir(parents=True)
    result = subprocess.run([sys.executable, str(ROOT / "prepare_data.py"),
                             "--data-root", str(dataset), "--mode", "from-edf"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert "No matching EDF files" in result.stderr


def test_synthetic_edf_reconstruction_and_standardization(tmp_path):
    from pyedflib import highlevel
    from neonatal_sleep.preprocessing import filter_and_resample_channel, read_edf

    dataset = tmp_path / "dataset"
    edf_dir = dataset / "data/source_edf"
    edf_dir.mkdir(parents=True)
    rate = 500
    time = np.arange(120 * rate) / rate
    signals = np.stack([20 * np.sin(2 * np.pi * (5 + index) * time) for index in range(8)])
    headers = highlevel.make_signal_headers(list(TARGET_CHANNELS), sample_frequency=rate,
                                           physical_min=-100, physical_max=100)
    header = highlevel.make_header(patientname="Synthetic subject", patientcode="SYNTHETIC")
    header["annotations"] = [[0, -1, "W-"], [60, -1, "-W"],
                             [60, -1, "N-"], [120, -1, "-N"]]
    path = edf_dir / "NSD0001.edf"
    highlevel.write_edf(str(path), signals, headers, header=header, digital=False)
    config_path = tmp_path / "config.json"
    config_path.write_text((ROOT / "config.json").read_text())
    labels = tmp_path / "labels.csv"
    # Deliberately consecutive retained-epoch coordinates. EDF annotations
    # must instead yield the matching W at 0 s and N at 60 s.
    labels.write_text("subject_id,epoch_index,label,epoch_start_seconds\n"
                      "NSD0001,0,W,0\nNSD0001,1,N,30\n")
    args = ("--config", str(config_path), "--data-root", str(dataset))
    run_cli("prepare_data.py", *args, "--mode", "from-edf", "--labels-csv", str(labels), cwd=tmp_path)
    loaded, _, _ = read_edf(path)
    processed = [filter_and_resample_channel(loaded[name], 500) / 1e6 for name in TARGET_CHANNELS]
    with np.load(tmp_path / "outputs/prepared/eeg_100hz/NSD0001.npz") as result:
        assert result["x"].shape == (2, 8, 3000)
        np.testing.assert_array_equal(result["y"], [0, 1])
        np.testing.assert_allclose(result["x"][1], np.stack([channel[6000:9000] for channel in processed]))
    run_cli("prepare_data.py", *args, "--mode", "sanitize-edf", cwd=tmp_path)
    _, standardized_headers, standardized = read_edf(tmp_path / "outputs/prepared/edf/NSD0001.edf")
    assert standardized["patientname"] != "Synthetic subject"
    assert standardized["patientcode"] != "SYNTHETIC"
    assert standardized["startdate"].year == 2000
    assert all(info["sample_frequency"] == 500 for info in standardized_headers.values())
