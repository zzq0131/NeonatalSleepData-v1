#!/usr/bin/env python3
"""Recompute the 86-dimensional aEEG feature tables and paper figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neonatal_sleep.config import load_config, resolve
from neonatal_sleep.statistics import extract_feature_table, make_plots, participant_stage_summary, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--data-root", type=Path, help="Extracted Figshare directory containing data/ and metadata/")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, data_root=args.data_root)
    root = Path(config["_project_root"])
    aeeg_dir = resolve(root, config["data"]["aeeg_epoch_dir"])
    output = Path(args.output_dir) if args.output_dir else resolve(root, config["outputs"]["statistics_root"])
    features, names = extract_feature_table(aeeg_dir)
    summary = participant_stage_summary(features, names)
    write_csv(output / "epoch_features_86d.csv", features)
    write_csv(output / "participant_stage_feature_medians.csv", summary)
    if not args.no_plots:
        paths = make_plots(features, summary, output / "figures")
        print("figures:")
        for path in paths:
            print(f"  {path}")
    print(f"epochs: {len(features)}; features: {len(names)}; participants: {features['subject_id'].nunique()}")
    print(f"tables: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

