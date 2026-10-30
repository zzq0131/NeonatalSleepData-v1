from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt


STAGE_TO_CODE = {"W": 0, "N": 1, "R": 2, "T": 3}
CODE_TO_STAGE = {value: key for key, value in STAGE_TO_CODE.items()}
TARGET_CHANNELS = (
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG T3-REF",
    "EEG T4-REF",
)
LEFT_PAIR = ("EEG C3-REF", "EEG P3-REF")
RIGHT_PAIR = ("EEG C4-REF", "EEG P4-REF")


def canonical_channel_name(value: str) -> str:
    text = " ".join(str(value).strip().upper().split())
    if text.startswith("EEG "):
        text = text[4:]
    if not text.endswith("-REF") and text in {name[4:-4] for name in TARGET_CHANNELS}:
        text += "-REF"
    return f"EEG {text}" if text in {name[4:] for name in TARGET_CHANNELS} else text


def parse_public_label_csv(path: str | Path, subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a public label table with subject_id, label and optional onset columns."""

    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {"subject_id", "label"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Label table is missing columns: {sorted(missing)}")
        for row in reader:
            if str(row["subject_id"]) == str(subject_id):
                rows.append(row)
    if not rows:
        raise ValueError(f"No public labels found for {subject_id}")
    rows.sort(key=lambda row: int(row.get("epoch_index", len(rows))))
    labels = np.asarray([parse_label(row["label"]) for row in rows], dtype=np.int32)
    if "epoch_start_seconds" in rows[0] and rows[0]["epoch_start_seconds"] not in {"", "NA", "nan"}:
        starts = np.asarray([float(row["epoch_start_seconds"]) for row in rows], dtype=float)
    else:
        starts = np.arange(labels.size, dtype=float) * 30.0
    return labels, starts


def parse_label(value: str | int | float) -> int:
    text = str(value).strip().upper()
    if text in STAGE_TO_CODE:
        return STAGE_TO_CODE[text]
    code = int(float(text))
    if code not in CODE_TO_STAGE:
        raise ValueError(f"label must be W/N/R/T or 0/1/2/3, got {value!r}")
    return code


def parse_stage_intervals(onsets: Iterable[float], descriptions: Iterable[str]) -> list[tuple[str, float, float]]:
    events = sorted((float(onset), str(description).strip().upper()) for onset, description in zip(onsets, descriptions))
    starts = {f"{stage}-" for stage in STAGE_TO_CODE}
    ends = {f"-{stage}" for stage in STAGE_TO_CODE}
    active: tuple[str, float] | None = None
    seen_start = False
    intervals: list[tuple[str, float, float]] = []
    for onset, description in events:
        if description in starts:
            if active is not None and onset > active[1]:
                intervals.append((active[0], active[1], onset))
            active = (description[0], onset)
            seen_start = True
        elif description in ends:
            if active is None and not seen_start:
                continue
            if active is not None and active[0] == description[1] and onset > active[1]:
                intervals.append((active[0], active[1], onset))
            active = None
    return intervals


def candidate_epoch_labels(intervals: list[tuple[str, float, float]], epoch_seconds: int = 30) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    starts: list[float] = []
    for stage, start, stop in intervals:
        count = int(np.floor((stop - start + 1e-9) / epoch_seconds))
        labels.extend([STAGE_TO_CODE[stage]] * count)
        starts.extend(start + index * epoch_seconds for index in range(count))
    return np.asarray(labels, dtype=np.int32), np.asarray(starts, dtype=float)


def extract_epoch_windows(
    processed: list[np.ndarray], starts_seconds: np.ndarray, *, sampling_rate_hz: int = 100
) -> np.ndarray:
    """Extract aligned 30-s windows at the selected EDF annotation starts."""

    starts = np.asarray(starts_seconds, dtype=float)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("aligned epoch starts must be a non-empty one-dimensional array")
    window_samples = int(30 * sampling_rate_hz)
    start_samples = np.rint(starts * sampling_rate_hz).astype(int)
    if np.any(start_samples < 0):
        raise ValueError("aligned epoch starts cannot be negative")
    windows: list[np.ndarray] = []
    for start in start_samples:
        stop = int(start + window_samples)
        channels = []
        for signal in processed:
            if stop > signal.size:
                raise ValueError(
                    f"aligned epoch window [{start / sampling_rate_hz:.3f}, "
                    f"{stop / sampling_rate_hz:.3f}) s exceeds the processed signal"
                )
            channels.append(signal[start:stop])
        windows.append(np.stack(channels, axis=0))
    return np.stack(windows, axis=0)


def realign_candidate_epochs(
    candidate_labels: np.ndarray,
    candidate_starts: np.ndarray,
    expected_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select an ordered candidate subsequence matching the public labels."""

    candidates = np.asarray(candidate_labels, dtype=np.int32)
    starts = np.asarray(candidate_starts, dtype=float)
    expected = np.asarray(expected_labels, dtype=np.int32)
    if candidates.shape != starts.shape:
        raise ValueError("candidate labels and starts must have equal shapes")
    if expected.size > candidates.size:
        raise ValueError("public labels exceed candidate EDF epochs")
    unreachable = np.iinfo(np.int64).max
    costs = np.full((candidates.size + 1, expected.size + 1), unreachable, dtype=np.int64)
    took = np.zeros_like(costs, dtype=bool)
    previous = np.full((candidates.size + 1, expected.size + 1, 2), -1, dtype=np.int64)
    costs[0, 0] = 0
    for i, candidate in enumerate(candidates):
        for j in range(expected.size + 1):
            if costs[i, j] == unreachable:
                continue
            if costs[i, j] < costs[i + 1, j]:
                costs[i + 1, j] = costs[i, j]
                previous[i + 1, j] = (i, j)
                took[i + 1, j] = False
            if j < expected.size and int(candidate) == int(expected[j]):
                value = costs[i, j] + i
                if value < costs[i + 1, j + 1]:
                    costs[i + 1, j + 1] = value
                    previous[i + 1, j + 1] = (i, j)
                    took[i + 1, j + 1] = True
    if costs[-1, -1] == unreachable:
        raise ValueError("EDF candidate epochs cannot be aligned to public labels")
    selected: list[int] = []
    i, j = candidates.size, expected.size
    while i or j:
        previous_i, previous_j = previous[i, j]
        if previous_i < 0:
            raise RuntimeError("invalid label alignment backtrack")
        if took[i, j]:
            selected.append(i - 1)
        i, j = int(previous_i), int(previous_j)
    selected.reverse()
    indices = np.asarray(selected, dtype=int)
    return expected.copy(), starts[indices]


def filter_and_resample_channel(
    signal: np.ndarray,
    source_rate_hz: int,
    target_rate_hz: int = 100,
    *,
    bandpass_hz: tuple[float, float] = (0.5, 35.0),
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if source_rate_hz <= 2 * bandpass_hz[1]:
        raise ValueError(f"source rate {source_rate_hz} Hz is too low for the requested band")
    sos = butter(4, bandpass_hz, btype="bandpass", fs=source_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, values)
    if notch_hz < source_rate_hz / 2:
        b, a = iirnotch(notch_hz, notch_q, fs=source_rate_hz)
        filtered = filtfilt(b, a, filtered)
    if source_rate_hz == target_rate_hz:
        return filtered
    divisor = np.gcd(int(source_rate_hz), int(target_rate_hz))
    return resample_poly(filtered, up=target_rate_hz // divisor, down=source_rate_hz // divisor)


def compute_epoch_aeeg(
    x: np.ndarray,
    channel_names: Iterable[str],
    *,
    source_unit: str = "V",
    sampling_rate_hz: int = 100,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3000:
        raise ValueError(f"expected x [N,C,3000], got {values.shape}")
    names = [str(value) for value in channel_names]
    canonical = {canonical_channel_name(name): index for index, name in enumerate(names)}
    try:
        c3, p3 = canonical[LEFT_PAIR[0]], canonical[LEFT_PAIR[1]]
        c4, p4 = canonical[RIGHT_PAIR[0]], canonical[RIGHT_PAIR[1]]
    except KeyError as exc:
        raise KeyError(f"missing aEEG source channel: {exc.args[0]}") from exc
    scale = 1e6 if source_unit.lower() in {"v", "volt", "volts"} else 1.0
    left = (values[:, c3, :] - values[:, p3, :]) * scale
    right = (values[:, c4, :] - values[:, p4, :]) * scale
    bipolar = np.stack([left, right], axis=1)
    bandpass = butter(4, [2.0, 15.0], btype="bandpass", fs=sampling_rate_hz, output="sos")
    filtered = sosfiltfilt(bandpass, bipolar, axis=-1)
    envelope = np.abs(filtered)
    smoother = butter(4, 0.5, btype="lowpass", fs=sampling_rate_hz, output="sos")
    envelope = sosfiltfilt(smoother, envelope, axis=-1)
    return np.maximum(resample_poly(envelope, up=1, down=sampling_rate_hz, axis=-1), 0.0).astype(np.float32)


def read_edf(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    try:
        from pyedflib import highlevel
    except ImportError as exc:
        raise RuntimeError("EDF processing requires pyedflib") from exc
    signals, headers, header = highlevel.read_edf(str(path), digital=False)
    by_name = {str(item["label"]): np.asarray(signal, dtype=float) for signal, item in zip(signals, headers)}
    metadata = {str(item["label"]): dict(item) for item in headers}
    return by_name, metadata, header


def anonymize_and_standardize_edf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_rate_hz: int = 500,
    target_channels: tuple[str, ...] = TARGET_CHANNELS,
) -> None:
    """Write a de-identified, standardized EDF copy without modifying input."""

    try:
        from pyedflib import highlevel
    except ImportError as exc:
        raise RuntimeError("EDF processing requires pyedflib") from exc
    signals, headers, header = highlevel.read_edf(str(input_path), digital=False)
    lookup = {canonical_channel_name(item["label"]): (signal, item) for signal, item in zip(signals, headers)}
    output_signals: list[np.ndarray] = []
    output_headers: list[dict] = []
    for target in target_channels:
        canonical = canonical_channel_name(target)
        if canonical not in lookup:
            raise KeyError(f"{input_path} is missing {target}")
        signal, source_header = lookup[canonical]
        source_rate = int(round(float(source_header["sample_frequency"])))
        processed = filter_and_resample_channel(signal, source_rate, target_rate_hz)
        output_signals.append(processed)
        out_header = deepcopy(source_header)
        out_header["label"] = target
        out_header["sample_frequency"] = float(target_rate_hz)
        output_headers.append(out_header)
    clean_header = deepcopy(header)
    for key in ("patientname", "patientcode", "patient_additional", "birthdate", "technician", "admincode"):
        if key in clean_header:
            clean_header[key] = ""
    # Do not expose the original calendar date. A fixed non-identifying date is used.
    if "startdate" in clean_header:
        clean_header["startdate"] = datetime(2000, 1, 1)
    public_markers = {f"{stage}-" for stage in STAGE_TO_CODE} | {f"-{stage}" for stage in STAGE_TO_CODE}
    clean_header["annotations"] = [
        annotation
        for annotation in header.get("annotations", [])
        if len(annotation) >= 3 and str(annotation[2]).strip().upper() in public_markers
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    highlevel.write_edf(str(target), output_signals, output_headers, header=clean_header, digital=False)
