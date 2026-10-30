from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .features import extract_86_features
from .io import load_aeeg_epoch, list_subject_files, subject_id_from_path


STAGES = ("W", "N", "R", "T")


def extract_feature_table(aeeg_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[pd.DataFrame] = []
    feature_names: list[str] | None = None
    for path in list_subject_files(aeeg_dir, ".npz"):
        subject_id = subject_id_from_path(path)
        record = load_aeeg_epoch(path)
        features, names = extract_86_features(record["aeeg"])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("feature names changed between subjects")
        frame = pd.DataFrame(features, columns=names)
        frame.insert(0, "stage", [STAGES[int(label)] for label in record["y"]])
        frame.insert(0, "label", record["y"])
        frame.insert(0, "epoch_index", np.arange(record["y"].size))
        frame.insert(0, "subject_id", subject_id)
        rows.append(frame)
    if not rows:
        raise ValueError("no aEEG files found")
    return pd.concat(rows, ignore_index=True), feature_names or []


def participant_stage_summary(features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    grouped = features.groupby(["subject_id", "label", "stage"], sort=True)
    summaries = grouped[feature_names].median().reset_index()
    counts = grouped.size().rename("n_epochs").reset_index()
    return counts.merge(summaries, on=["subject_id", "label", "stage"], validate="one_to_one")


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")


def make_plots(features: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plot generation requires matplotlib") from exc
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    counts = features.groupby("stage").size().reindex(STAGES, fill_value=0)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(counts.index, counts.values, color=["#0072B2", "#009E73", "#D55E00", "#CC79A7"])
    axis.set_xlabel("Sleep stage")
    axis.set_ylabel("Epoch count")
    axis.set_title("Labelled 30-s epoch distribution")
    figure.tight_layout()
    path = target / "label_distribution.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    generated.append(str(path))

    selected = ["left_median", "right_median", "left_p90_p10", "bilateral_pearson_correlation"]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, feature in zip(axes.flat, selected):
        if feature not in summary:
            axis.set_visible(False)
            continue
        data = [summary.loc[summary["stage"] == stage, feature].dropna().to_numpy() for stage in STAGES]
        axis.boxplot(data, labels=STAGES, showfliers=False)
        axis.set_title(feature)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Participant-stage median aEEG features")
    figure.tight_layout()
    path = target / "participant_stage_features.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    generated.append(str(path))
    return generated

