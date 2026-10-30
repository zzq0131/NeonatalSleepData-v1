#!/usr/bin/env python3
"""Run subject-independent Random Forest and CNN baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neonatal_sleep.baselines import run_cnn, run_random_forest
from neonatal_sleep.config import load_config, resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--data-root", type=Path, help="Extracted Figshare directory containing data/ and metadata/")
    parser.add_argument("--model", choices=("random_forest", "cnn", "both"), default="both")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config, data_root=args.data_root)
    root = Path(config["_project_root"])
    aeeg_dir = resolve(root, config["data"]["aeeg_epoch_dir"])
    splits = resolve(root, config["splits_file"])
    output = Path(args.output_dir) if args.output_dir else resolve(root, config["outputs"]["baselines_root"])
    if args.model in {"random_forest", "both"}:
        metrics = run_random_forest(aeeg_dir, splits, output, config["baselines"])
        print(f"random forest macro-F1: {metrics['macro_f1']:.4f}")
    if args.model in {"cnn", "both"}:
        metrics = run_cnn(aeeg_dir, splits, output, config["baselines"])
        print(f"CNN macro-F1: {metrics['macro_f1']:.4f}")
    print(f"baseline outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

