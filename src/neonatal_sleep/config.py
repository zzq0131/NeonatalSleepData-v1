from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path, *, data_root: str | Path | None = None) -> dict:
    config_path = Path(path).resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config.json must contain a JSON object")
    value["_config_path"] = str(config_path)
    value["_project_root"] = str(config_path.parent)
    if data_root is not None:
        dataset = Path(data_root).expanduser().resolve()
        if not (dataset / "data").is_dir():
            raise FileNotFoundError(f"--data-root must contain a data/ directory: {dataset}")
        for key in ("edf_dir", "eeg100_dir", "aeeg_epoch_dir"):
            value["data"][key] = str(resolve(dataset, value["data"][key]))
        value["_data_root"] = str(dataset)
    return value


def resolve(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path

