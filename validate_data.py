#!/usr/bin/env python3
"""Validate file structure, arrays, labels, timing metadata and amplitudes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neonatal_sleep.config import load_config, resolve
from neonatal_sleep.io import write_json
from neonatal_sleep.validation import validate_release, write_participant_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--data-root", type=Path, help="Extracted Figshare directory containing data/ and metadata/")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    config = load_config(args.config, data_root=args.data_root)
    root = Path(config["_project_root"])
    result = validate_release(root, config)
    output_root = resolve(root, config["outputs"]["validation_root"])
    output_json = Path(args.output_json) if args.output_json else output_root / "validation_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else output_root / "participant_validation.csv"
    write_json(output_json, {key: value for key, value in result.items() if key != "participants"})
    write_participant_csv(output_csv, result["participants"])
    print(f"validated {result['eeg100_files']} EEG100 files and {result['aeeg_epoch_files']} aEEG files")
    print(f"epochs: {result['labelled_epoch_count']}; stage counts: {result['stage_epoch_counts']}")
    print(f"summary: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

