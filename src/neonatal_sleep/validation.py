from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .io import STAGE_NAMES, load_aeeg_epoch, load_eeg100, sha256, subject_id_from_path


def validate_release(root: str | Path, config: dict) -> dict:
    root = Path(root)
    data = config["data"]
    eeg_dir = root / data["eeg100_dir"]
    aeeg_dir = root / data["aeeg_epoch_dir"]
    edf_dir = root / data["edf_dir"]
    expected = int(config["release"]["expected_subjects"])
    eeg_files = sorted(eeg_dir.glob(config["data"].get("eeg100_glob", "NSD*.npz")))
    aeeg_files = sorted(aeeg_dir.glob(config["data"].get("aeeg_glob", "NSD*.npz")))
    edf_files = sorted(edf_dir.glob(config["data"].get("edf_glob", "NSD*.edf"))) if edf_dir.exists() else []
    if len(eeg_files) != expected or len(aeeg_files) != expected:
        raise ValueError(f"expected {expected} EEG and aEEG files, got {len(eeg_files)} and {len(aeeg_files)}")
    aeeg_by_id = {subject_id_from_path(path): path for path in aeeg_files}
    counts = np.zeros(4, dtype=np.int64)
    amplitudes: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for eeg_path in eeg_files:
        subject_id = subject_id_from_path(eeg_path)
        if subject_id not in aeeg_by_id:
            raise ValueError(f"missing aEEG file for {subject_id}")
        eeg = load_eeg100(eeg_path)
        aeeg = load_aeeg_epoch(aeeg_by_id[subject_id])
        if not np.array_equal(eeg["y"], aeeg["y"]):
            raise ValueError(f"label mismatch between EEG and aEEG for {subject_id}")
        counts += np.bincount(eeg["y"], minlength=4)
        amplitudes.append(aeeg["aeeg"].reshape(-1))
        rows.append({
            "subject_id": subject_id,
            "eeg_file": eeg_path.name,
            "aeeg_file": aeeg_by_id[subject_id].name,
            "epoch_count": int(eeg["y"].size),
            "W": int(np.sum(eeg["y"] == 0)),
            "N": int(np.sum(eeg["y"] == 1)),
            "R": int(np.sum(eeg["y"] == 2)),
            "T": int(np.sum(eeg["y"] == 3)),
            "eeg_sha256": sha256(eeg_path),
            "aeeg_sha256": sha256(aeeg_by_id[subject_id]),
        })
    flat = np.concatenate(amplitudes)
    result = {
        "passed": True,
        "expected_subjects": expected,
        "eeg100_files": len(eeg_files),
        "aeeg_epoch_files": len(aeeg_files),
        "edf_files": len(edf_files),
        "labelled_epoch_count": int(counts.sum()),
        "stage_epoch_counts": {STAGE_NAMES[i]: int(value) for i, value in enumerate(counts)},
        "stage_proportions": {STAGE_NAMES[i]: float(value / counts.sum()) for i, value in enumerate(counts)},
        "aeeg_amplitude_uv": {
            "median": float(np.median(flat)),
            "p99": float(np.percentile(flat, 99)),
            "maximum": float(np.max(flat)),
            "count_gt_100_uv": int(np.sum(flat > 100.0)),
            "fraction_gt_100_uv": float(np.mean(flat > 100.0)),
        },
        "participants": rows,
    }
    return result


def write_participant_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no validation rows")
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

