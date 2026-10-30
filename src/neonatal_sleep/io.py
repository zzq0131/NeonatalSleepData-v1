from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


STAGE_NAMES = {0: "W", 1: "N", 2: "R", 3: "T"}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_id_from_path(path: str | Path) -> str:
    name = Path(path).stem
    if name.endswith("_eeg100"):
        return name.removesuffix("_eeg100")
    if name.endswith("_aeeg_epoch"):
        return name.removesuffix("_aeeg_epoch")
    return name


def list_subject_files(directory: str | Path, suffix: str) -> list[Path]:
    paths = sorted(Path(directory).glob(f"NSD*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No files matching NSD*{suffix} in {directory}")
    return paths


def load_eeg100(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "y", "sampling_rate", "channel_names"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        record = {key: archive[key] for key in required}
    x = np.asarray(record["x"])
    y = np.asarray(record["y"], dtype=np.int32)
    if x.ndim != 3 or x.shape[0] != y.size or x.shape[2] != 3000:
        raise ValueError(f"Invalid EEG100 shape in {path}: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError(f"Non-finite values or invalid labels in {path}")
    rate = int(round(float(np.asarray(record["sampling_rate"]).item())))
    if rate != 100:
        raise ValueError(f"Expected 100-Hz EEG in {path}, got {rate} Hz")
    channels = np.asarray(record["channel_names"]).astype(str)
    if channels.ndim != 1 or channels.size != x.shape[1]:
        raise ValueError(f"Channel metadata does not match x in {path}")
    return {"x": x, "y": y, "sampling_rate": np.asarray(rate), "channel_names": channels}


def load_aeeg_epoch(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "aeeg",
            "y",
            "source_sampling_rate_hz",
            "aeeg_sampling_rate_hz",
            "source_window_seconds",
            "bipolar_channel_names",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        record = {key: archive[key] for key in required}
    aeeg = np.asarray(record["aeeg"])
    y = np.asarray(record["y"], dtype=np.int32)
    if aeeg.ndim != 3 or aeeg.shape[0] != y.size or aeeg.shape[1:] != (2, 30):
        raise ValueError(f"Invalid aEEG shape in {path}: {aeeg.shape}")
    if not np.isfinite(aeeg).all() or not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError(f"Non-finite values or invalid labels in {path}")
    if int(round(float(np.asarray(record["aeeg_sampling_rate_hz"]).item()))) != 1:
        raise ValueError(f"Expected 1-Hz aEEG in {path}")
    if int(round(float(np.asarray(record["source_window_seconds"]).item()))) != 30:
        raise ValueError(f"Expected 30-s windows in {path}")
    return {**record, "aeeg": aeeg, "y": y}


def write_tsv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

