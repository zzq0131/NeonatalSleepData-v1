#!/usr/bin/env python3
"""Public data-preparation entry point.

Examples:
    python prepare_data.py --mode from-existing
    python prepare_data.py --mode from-edf --labels-csv metadata/labels.csv
    python prepare_data.py --mode sanitize-edf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neonatal_sleep.config import load_config, resolve
from neonatal_sleep.io import load_eeg100, list_subject_files, subject_id_from_path
from neonatal_sleep.preprocessing import (
    TARGET_CHANNELS,
    anonymize_and_standardize_edf,
    candidate_epoch_labels,
    canonical_channel_name,
    compute_epoch_aeeg,
    extract_epoch_windows,
    filter_and_resample_channel,
    parse_public_label_csv,
    parse_stage_intervals,
    realign_candidate_epochs,
    read_edf,
)


def _output_names(subject_id: str) -> tuple[str, str]:
    return f"{subject_id}.npz", f"{subject_id}.npz"


def save_pair(output_root: Path, subject_id: str, x: np.ndarray, y: np.ndarray, aeeg: np.ndarray) -> None:
    eeg_dir = output_root / "eeg_100hz"
    aeeg_dir = output_root / "aeeg_epoch"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    aeeg_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        eeg_dir / _output_names(subject_id)[0],
        x=x.astype(np.float64),
        y=y.astype(np.int32),
        sampling_rate=np.asarray(100, dtype=np.int32),
        channel_names=np.asarray(TARGET_CHANNELS),
    )
    np.savez_compressed(
        aeeg_dir / _output_names(subject_id)[1],
        aeeg=aeeg.astype(np.float32),
        y=y.astype(np.int32),
        source_sampling_rate_hz=np.asarray(100.0, dtype=np.float32),
        aeeg_sampling_rate_hz=np.asarray(1, dtype=np.int32),
        source_window_seconds=np.asarray(30, dtype=np.int32),
        bipolar_channel_names=np.asarray(["EEG C3-REF - EEG P3-REF", "EEG C4-REF - EEG P4-REF"]),
    )


def from_existing(config: dict, labels_csv: Path | None, subject_filter: str | None) -> None:
    root = Path(config["_project_root"])
    source = resolve(root, config["data"]["eeg100_dir"])
    output_root = resolve(root, config["outputs"]["prepared_root"])
    paths = list_subject_files(source, ".npz")
    if subject_filter and not any(subject_id_from_path(path) == subject_filter for path in paths):
        raise FileNotFoundError(f"No EEG100 file for {subject_filter} in {source}")
    for path in paths:
        subject_id = subject_id_from_path(path)
        if subject_filter and subject_id != subject_filter:
            continue
        record = load_eeg100(path)
        labels = record["y"].astype(np.int32)
        if labels_csv is not None:
            labels, _ = parse_public_label_csv(labels_csv, subject_id)
            if labels.size != record["x"].shape[0]:
                raise ValueError(f"public labels do not match {subject_id}: {labels.size} != {record['x'].shape[0]}")
        aeeg = compute_epoch_aeeg(
            record["x"],
            record["channel_names"],
            source_unit=config["preprocessing"].get("eeg100_values_unit", "V"),
            sampling_rate_hz=100,
        )
        save_pair(output_root, subject_id, record["x"], labels, aeeg)
        print(f"prepared {subject_id}: {labels.size} epochs")


def _edf_annotations(header: dict) -> tuple[np.ndarray, np.ndarray]:
    annotations = header.get("annotations", [])
    onsets: list[float] = []
    descriptions: list[str] = []
    for annotation in annotations:
        if len(annotation) < 3:
            continue
        onsets.append(float(annotation[0]))
        descriptions.append(str(annotation[2]))
    return np.asarray(onsets), np.asarray(descriptions)


def _edf_paths(source: Path, config: dict, subject_filter: str | None) -> list[Path]:
    paths = sorted(source.glob(config["data"].get("edf_glob", "NSD*.edf")))
    if subject_filter:
        paths = [path for path in paths if path.stem == subject_filter]
    if not paths:
        raise FileNotFoundError(f"No matching EDF files in {source}; subject={subject_filter or 'all'}")
    return paths


def from_edf(config: dict, labels_csv: Path | None, subject_filter: str | None) -> None:
    root = Path(config["_project_root"])
    source = resolve(root, config["data"]["edf_dir"])
    output_root = resolve(root, config["outputs"]["prepared_root"])
    for path in _edf_paths(source, config, subject_filter):
        subject_id = path.stem
        if subject_filter and subject_id != subject_filter:
            continue
        signals, headers, header = read_edf(path)
        lookup = {canonical_channel_name(label): signal for label, signal in signals.items()}
        processed: list[np.ndarray] = []
        for target in TARGET_CHANNELS:
            key = canonical_channel_name(target)
            if key not in lookup:
                raise KeyError(f"{path.name} is missing {target}")
            source_rate = int(round(float(headers[next(label for label in headers if canonical_channel_name(label) == key)]["sample_frequency"])))
            values_uv = filter_and_resample_channel(
                lookup[key],
                source_rate,
                target_rate_hz=100,
                bandpass_hz=tuple(config["preprocessing"]["edf_bandpass_hz"]),
                notch_hz=float(config["preprocessing"]["edf_notch_hz"]),
                notch_q=float(config["preprocessing"]["edf_notch_q"]),
            )
            processed.append(values_uv / 1e6)  # public EEG100 stores volts
        if labels_csv is not None:
            labels, starts = parse_public_label_csv(labels_csv, subject_id)
            onsets, descriptions = _edf_annotations(header)
            if onsets.size:
                candidates, candidate_starts = candidate_epoch_labels(parse_stage_intervals(onsets, descriptions))
                labels, starts = realign_candidate_epochs(candidates, candidate_starts, labels)
        else:
            onsets, descriptions = _edf_annotations(header)
            candidates, starts = candidate_epoch_labels(parse_stage_intervals(onsets, descriptions))
            labels = candidates
        if labels.size == 0:
            raise ValueError(f"no labels or stage annotations found for {subject_id}")
        x = extract_epoch_windows(processed, starts, sampling_rate_hz=100)
        aeeg = compute_epoch_aeeg(x, TARGET_CHANNELS, source_unit="V", sampling_rate_hz=100)
        save_pair(output_root, subject_id, x, labels, aeeg)
        print(f"prepared {subject_id} from EDF: {labels.size} epochs")


def sanitize_edf(config: dict, subject_filter: str | None) -> None:
    root = Path(config["_project_root"])
    source = resolve(root, config["data"]["edf_dir"])
    output = resolve(root, config["outputs"]["sanitized_edf_dir"])
    for path in _edf_paths(source, config, subject_filter):
        if subject_filter and path.stem != subject_filter:
            continue
        anonymize_and_standardize_edf(
            path,
            output / path.name,
            target_rate_hz=int(config["preprocessing"]["edf_target_rate_hz"]),
        )
        print(f"sanitized {path.stem}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--data-root", type=Path, help="Extracted Figshare directory containing data/ and metadata/")
    parser.add_argument("--mode", choices=("from-existing", "from-edf", "sanitize-edf"), default="from-existing")
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--subject")
    args = parser.parse_args()
    config = load_config(args.config, data_root=args.data_root)
    labels = args.labels_csv.resolve() if args.labels_csv else None
    if args.mode == "from-existing":
        from_existing(config, labels, args.subject)
    elif args.mode == "from-edf":
        from_edf(config, labels, args.subject)
    else:
        sanitize_edf(config, args.subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
