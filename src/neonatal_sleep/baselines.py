from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .features import extract_86_features
from .io import list_subject_files, load_aeeg_epoch, subject_id_from_path


STAGES = ("W", "N", "R", "T")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_splits(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if "outer_folds" not in value:
        raise ValueError("splits.json must contain outer_folds")
    return value


def load_dataset(
    aeeg_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    signal_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    epoch_indices: list[np.ndarray] = []
    for path in list_subject_files(aeeg_dir, ".npz"):
        record = load_aeeg_epoch(path)
        subject = subject_id_from_path(path)
        features, _ = extract_86_features(record["aeeg"])
        feature_rows.append(features)
        signal_rows.append(np.asarray(record["aeeg"], dtype=np.float32))
        labels.append(record["y"])
        subjects.append(np.full(record["y"].size, subject, dtype=object))
        epoch_indices.append(np.arange(record["y"].size, dtype=int))
    if not feature_rows:
        raise ValueError("no aEEG files found")
    return (
        np.vstack(feature_rows),
        np.concatenate(signal_rows),
        np.concatenate(labels),
        np.concatenate(subjects),
        np.concatenate(epoch_indices),
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3])),
        "confusion_matrix": matrix.tolist(),
    }


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray, seed: int, repetitions: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects)
    scores = {name: [] for name in ("balanced_accuracy", "macro_f1", "cohen_kappa")}
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        mask = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
        values = _metrics(y_true[mask], y_pred[mask])
        for name in scores:
            scores[name].append(values[name])
    return {
        name: {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}
        for name, values in scores.items()
    }


def _rf_model(params: dict[str, Any], seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 500)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 5)),
        max_features=params.get("max_features", "sqrt"),
        class_weight=params.get("class_weight", "balanced_subsample"),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=seed,
    )


class _TrainOnlyWinsorizer:
    def __init__(self, lower: float = 0.001, upper: float = 0.999) -> None:
        self.lower_quantile = lower
        self.upper_quantile = upper

    def fit(self, values: np.ndarray) -> "_TrainOnlyWinsorizer":
        self.lower_ = np.quantile(values, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(values, self.upper_quantile, axis=0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.lower_, self.upper_)


class _TrainOnlyCorrelationFilter:
    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = threshold

    def fit(self, values: np.ndarray) -> "_TrainOnlyCorrelationFilter":
        active = np.flatnonzero(np.std(values, axis=0) > 1e-12)
        keep = np.zeros(values.shape[1], dtype=bool)
        correlation = np.corrcoef(values[:, active], rowvar=False) if active.size else np.empty((0, 0))
        selected: list[int] = []
        for local, feature in enumerate(active):
            if all(abs(float(correlation[local, previous])) <= self.threshold for previous in selected):
                keep[feature] = True
                selected.append(local)
        if not keep.any():
            raise ValueError("correlation filter removed every feature")
        self.support_ = keep
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return values[:, self.support_]


def _fit_rf_preprocessing(train_x: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, Any]:
    transforms: list[Any] = []
    quantiles = config.get("winsorize_quantiles", [0.001, 0.999])
    if quantiles is not None:
        transform = _TrainOnlyWinsorizer(float(quantiles[0]), float(quantiles[1])).fit(train_x)
        train_x = transform.transform(train_x)
        transforms.append(transform)
    threshold = config.get("correlation_threshold", 0.98)
    if threshold is not None:
        transform = _TrainOnlyCorrelationFilter(float(threshold)).fit(train_x)
        train_x = transform.transform(train_x)
        transforms.append(transform)
    return train_x, transforms


def _transform_rf(values: np.ndarray, transforms: list[Any]) -> np.ndarray:
    output = values
    for transform in transforms:
        output = transform.transform(output)
    return output


def _choose_rf_params(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    train_subjects: list[str],
    inner_folds: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    seed: int,
    preprocessing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_set = set(train_subjects)
    best_score = -np.inf
    best = dict(grid[0])
    preprocessing_config = preprocessing_config or {}
    for params in grid:
        fit_config = {**preprocessing_config, **params}
        fold_scores = []
        for inner in inner_folds:
            test = train_set.intersection(inner["test_subjects"])
            train = train_set.difference(test)
            train_mask = np.isin(subjects, list(train))
            test_mask = np.isin(subjects, list(test))
            if not np.any(train_mask) or not np.any(test_mask):
                continue
            train_x, transforms = _fit_rf_preprocessing(x[train_mask], fit_config)
            test_x = _transform_rf(x[test_mask], transforms)
            model = _rf_model(params, seed)
            model.fit(train_x, y[train_mask])
            fold_scores.append(_metrics(y[test_mask], model.predict(test_x))["macro_f1"])
        score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if score > best_score:
            best_score, best = score, dict(params)
    return best


def run_random_forest(
    aeeg_dir: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("random forest baseline requires scikit-learn") from exc
    seed = int(config.get("seed", 20260711))
    seed_everything(seed)
    x, _, y, subjects, epoch_indices = load_dataset(aeeg_dir)
    splits = load_splits(splits_path)
    outer = [{"fold": int(item["fold"]), "test_subjects": set(item["test_subjects"])} for item in splits["outer_folds"]]
    inner = splits.get("inner_folds", {})
    rf_config = config.get("rf", {})
    grid = rf_config.get("parameter_grid", [{}])
    predictions: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for fold in outer:
        test_subjects = fold["test_subjects"]
        train_subjects = sorted(set(subjects.tolist()).difference(test_subjects))
        params = _choose_rf_params(
            x,
            y,
            subjects,
            train_subjects,
            inner.get(str(fold["fold"]), []),
            grid,
            seed,
            preprocessing_config=rf_config,
        )
        train_mask = np.isin(subjects, train_subjects)
        test_mask = np.isin(subjects, list(test_subjects))
        train_x, transforms = _fit_rf_preprocessing(
            x[train_mask], {**rf_config, **params}
        )
        test_x = _transform_rf(x[test_mask], transforms)
        model = _rf_model(params, seed)
        model.fit(train_x, y[train_mask])
        predicted = model.predict(test_x).astype(int)
        probabilities = model.predict_proba(test_x)
        classes = np.asarray(model.classes_, dtype=int)
        aligned = np.zeros((predicted.size, 4), dtype=float)
        for index, label in enumerate(classes):
            aligned[:, label] = probabilities[:, index]
        for index, position in enumerate(np.flatnonzero(test_mask)):
            predictions.append({
                "outer_fold": fold["fold"],
                "subject_id": str(subjects[position]),
                "epoch_index": int(epoch_indices[position]),
                "y_true": int(y[position]),
                "y_pred": int(predicted[index]),
                **{f"prob_{STAGES[j]}": float(aligned[index, j]) for j in range(4)},
            })
        selected[str(fold["fold"])] = {
            "model_parameters": params,
            "winsorize_quantiles": rf_config.get("winsorize_quantiles", [0.001, 0.999]),
            "correlation_threshold": rf_config.get("correlation_threshold", 0.98),
        }
    return _write_result(predictions, selected, output_dir, "random_forest", seed, int(config.get("bootstrap_repetitions", 2000)))


def _torch_device(torch: Any, requested: str = "auto") -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _transform_cnn_input(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, 30):
        raise ValueError(f"aEEG CNN input must have shape [N,2,30], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("aEEG CNN input must be finite")
    mode = str(mode).lower()
    if mode in {"log1p", "log1p_z"}:
        if np.any(values < 0):
            raise ValueError("log1p aEEG input requires nonnegative values")
        values = np.log1p(values)
    elif mode not in {"raw", "raw_z"}:
        raise ValueError("CNN input_transform must be raw, raw_z, log1p or log1p_z")
    return values.astype(np.float32, copy=False)


def _prepare_cnn_split(
    train_x: np.ndarray, other: list[np.ndarray], mode: str
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    transformed_train = _transform_cnn_input(train_x, mode)
    transformed_other = [_transform_cnn_input(values, mode) for values in other]
    if str(mode).lower().endswith("_z"):
        mean = transformed_train.mean(axis=(0, 2), keepdims=True)
        std = transformed_train.std(axis=(0, 2), keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        transformed_train = (transformed_train - mean) / std
        transformed_other = [(values - mean) / std for values in transformed_other]
    else:
        mean = np.zeros((1, train_x.shape[1], 1), dtype=np.float32)
        std = np.ones((1, train_x.shape[1], 1), dtype=np.float32)
    return (
        transformed_train.astype(np.float32),
        [values.astype(np.float32) for values in transformed_other],
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def _MultiscaleBlock(dropout: float = 0.2):
    """Build the released compact 3/5/7-kernel multiscale CNN."""
    import torch
    from torch import nn

    class Multiscale(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.branches = nn.ModuleList(
                [nn.Conv1d(2, 16, kernel_size=k, padding=k // 2) for k in (3, 5, 7)]
            )
            self.fusion_norm = nn.BatchNorm1d(48)
            self.fusion_activation = nn.GELU()
            self.residual = nn.Sequential(
                nn.Conv1d(48, 48, kernel_size=3, padding=1),
                nn.BatchNorm1d(48),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(96, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 4)
            )

        def forward(self, values):
            fused = torch.cat([branch(values) for branch in self.branches], dim=1)
            fused = self.fusion_activation(self.fusion_norm(fused))
            fused = fused + self.residual(fused)
            pooled = torch.cat([fused.mean(dim=2), fused.amax(dim=2)], dim=1)
            return self.classifier(pooled)

    return Multiscale()


class _SmallAeegCNN:
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        seed: int,
        dropout: float = 0.2,
        batch_size: int = 128,
        patience: int = 15,
        class_weight_strategy: str = "effective_number",
        effective_number_beta: float = 0.99,
        gradient_clip_norm: float = 5.0,
        device: str = "auto",
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CNN baseline requires PyTorch") from exc
        seed_everything(seed)
        self.torch = torch
        self.seed = int(seed)
        self.device = _torch_device(torch, device)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.class_weight_strategy = class_weight_strategy
        self.effective_number_beta = float(effective_number_beta)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.model = _MultiscaleBlock(dropout=dropout).to(self.device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)

    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.asarray(y, dtype=int), minlength=4).astype(float)
        if self.class_weight_strategy == "effective_number":
            weights = (1.0 - self.effective_number_beta) / (
                1.0 - np.power(self.effective_number_beta, counts)
            )
            weights[counts == 0] = 0.0
            present = weights > 0
            if np.any(present):
                weights[present] /= weights[present].mean()
        elif self.class_weight_strategy in {"inverse", "inverse_frequency"}:
            weights = counts.sum() / (4 * np.maximum(counts, 1.0))
        elif self.class_weight_strategy in {"none", "unweighted"}:
            weights = np.ones(4, dtype=float)
        else:
            raise ValueError(f"unknown CNN class-weight strategy: {self.class_weight_strategy}")
        return weights.astype(np.float32)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_x: np.ndarray | None = None,
        validation_y: np.ndarray | None = None,
        epochs: int | None = None,
    ) -> list[dict[str, float]]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        values = np.asarray(x, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int64)
        if values.ndim != 3 or values.shape[1:] != (2, 30):
            raise ValueError(f"aEEG CNN input must have shape [N,2,30], got {values.shape}")
        if labels.shape != (len(values),):
            raise ValueError("CNN labels must align with input epochs")
        if (validation_x is None) != (validation_y is None):
            raise ValueError("validation_x and validation_y must be supplied together")
        validation_values = None
        validation_labels = None
        if validation_x is not None and validation_y is not None:
            validation_values = np.asarray(validation_x, dtype=np.float32)
            validation_labels = np.asarray(validation_y, dtype=np.int64)
            if validation_values.ndim != 3 or validation_values.shape[1:] != (2, 30):
                raise ValueError("invalid CNN validation shape")
            if validation_labels.shape != (len(validation_values),):
                raise ValueError("CNN validation labels must align with validation input")
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(self._class_weights(labels), dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loader = DataLoader(
            TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        validation_tensor = (
            torch.from_numpy(validation_values).to(self.device)
            if validation_values is not None
            else None
        )
        validation_target = (
            torch.from_numpy(validation_labels).to(self.device)
            if validation_labels is not None
            else None
        )
        history: list[dict[str, float]] = []
        best_state: dict[str, Any] | None = None
        best_loss = float("inf")
        stale = 0
        max_epochs = int(self.epochs if epochs is None else epochs)
        for current_epoch in range(max_epochs):
            self.model.train()
            losses = []
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(inputs), targets)
                loss.backward()
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            row = {"epoch": float(current_epoch + 1), "train_loss": float(np.mean(losses))}
            if validation_tensor is not None and validation_target is not None:
                self.model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        loss_fn(self.model(validation_tensor), validation_target).detach().cpu()
                    )
                row["validation_loss"] = validation_loss
                if validation_loss < best_loss - 1e-6:
                    best_loss = validation_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
            history.append(row)
            if validation_tensor is not None and stale >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict_proba(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (2, 30):
            raise ValueError(f"aEEG CNN input must have shape [N,2,30], got {values.shape}")
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size, shuffle=False)
        outputs = []
        with torch.no_grad():
            for (batch,) in loader:
                outputs.append(torch.softmax(self.model(batch.to(self.device)), dim=1).cpu().numpy())
        return np.vstack(outputs)


def _new_cnn(config: dict[str, Any], seed: int) -> _SmallAeegCNN:
    architecture = str(config.get("architecture", "multiscale")).lower()
    if architecture not in {"multiscale", "c1", "multiscale_aeeg_cnn"}:
        raise ValueError("the released CNN baseline uses the multiscale architecture")
    return _SmallAeegCNN(
        learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        epochs=int(config.get("epochs", 30)),
        seed=seed,
        dropout=float(config.get("dropout", 0.2)),
        batch_size=int(config.get("batch_size", 128)),
        patience=int(config.get("patience", 15)),
        class_weight_strategy=str(config.get("class_weight_strategy", "effective_number")),
        effective_number_beta=float(config.get("effective_number_beta", 0.99)),
        gradient_clip_norm=float(config.get("gradient_clip_norm", 5.0)),
        device=str(config.get("device", "auto")),
    )


def _choose_cnn_params(
    signals: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    train_subjects: list[str],
    inner_folds: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    base_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Select CNN settings using subject-disjoint inner validation folds."""

    train_set = set(train_subjects)
    best_score = -np.inf
    best = dict(grid[0])
    for params in grid:
        current = {**base_config, **params}
        scores: list[float] = []
        for inner in inner_folds:
            validation_subjects = train_set.intersection(inner["test_subjects"])
            fit_subjects = train_set.difference(validation_subjects)
            fit_mask = np.isin(subjects, list(fit_subjects))
            validation_mask = np.isin(subjects, list(validation_subjects))
            if not np.any(fit_mask) or not np.any(validation_mask):
                continue
            train_x, other, _, _ = _prepare_cnn_split(
                signals[fit_mask], [signals[validation_mask]], str(current.get("input_transform", "raw_z"))
            )
            model = _new_cnn(current, seed)
            model.fit(
                train_x,
                y[fit_mask],
                validation_x=other[0],
                validation_y=y[validation_mask],
            )
            score = _metrics(
                y[validation_mask],
                model.predict_proba(other[0], int(current.get("inference_batch_size", 512))).argmax(axis=1),
            )["macro_f1"]
            scores.append(score)
        score = float(np.mean(scores)) if scores else -np.inf
        if score > best_score:
            best_score, best = score, dict(params)
    return best


def run_cnn(
    aeeg_dir: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("seed", 20260711))
    _, signals, y, subjects, epoch_indices = load_dataset(aeeg_dir)
    splits = load_splits(splits_path)
    cnn_config = config.get("cnn", {})
    grid = cnn_config.get("parameter_grid", [{}])
    predictions: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for fold in splits["outer_folds"]:
        fold_id = int(fold["fold"])
        test_subjects = set(fold["test_subjects"])
        train_subjects = sorted(set(subjects.tolist()).difference(test_subjects))
        params = _choose_cnn_params(
            signals,
            y,
            subjects,
            train_subjects,
            splits.get("inner_folds", {}).get(str(fold_id), []),
            grid,
            cnn_config,
            seed + fold_id,
        )
        current = {**cnn_config, **params}
        train_mask = np.isin(subjects, train_subjects)
        test_mask = np.isin(subjects, list(test_subjects))
        input_transform = str(current.get("input_transform", "raw_z"))
        selection_protocol = str(current.get("selection_protocol", "single_validation"))
        inner_rows = splits.get("inner_folds", {}).get(str(fold_id), [])
        if selection_protocol not in {"all_inner_refit", "single_validation"}:
            raise ValueError("CNN selection_protocol must be single_validation or all_inner_refit")
        seeds = tuple(int(value) for value in current.get("seeds", [seed + fold_id]))
        if not seeds:
            raise ValueError("CNN seeds must not be empty")

        refit_epochs = int(current.get("epochs", 30))
        if selection_protocol == "all_inner_refit":
            histories: list[list[dict[str, float]]] = []
            for inner_index, inner in enumerate(inner_rows):
                validation_subjects = set(train_subjects).intersection(inner["test_subjects"])
                fit_subjects = set(train_subjects).difference(validation_subjects)
                fit_mask = np.isin(subjects, list(fit_subjects))
                validation_mask = np.isin(subjects, list(validation_subjects))
                train_x, other, _, _ = _prepare_cnn_split(
                    signals[fit_mask], [signals[validation_mask]], input_transform
                )
                model = _new_cnn(current, seed + fold_id * 100 + inner_index)
                histories.append(
                    model.fit(
                        train_x,
                        y[fit_mask],
                        validation_x=other[0],
                        validation_y=y[validation_mask],
                    )
                )
            best_epochs = [
                min(
                    [row for row in history if "validation_loss" in row],
                    key=lambda row: row["validation_loss"],
                )["epoch"]
                for history in histories
                if any("validation_loss" in row for row in history)
            ]
            if best_epochs:
                refit_epochs = max(1, int(round(float(np.median(best_epochs)))))
            train_x, other, channel_mean, channel_std = _prepare_cnn_split(
                signals[train_mask], [signals[test_mask]], input_transform
            )
            test_x = other[0]
            validation_x = validation_y = None
        else:
            if not inner_rows:
                raise ValueError("single_validation requires at least one inner fold")
            validation_subjects = set(train_subjects).intersection(inner_rows[0]["test_subjects"])
            fit_subjects = set(train_subjects).difference(validation_subjects)
            fit_mask = np.isin(subjects, list(fit_subjects))
            validation_mask = np.isin(subjects, list(validation_subjects))
            train_x, other, channel_mean, channel_std = _prepare_cnn_split(
                signals[fit_mask], [signals[validation_mask], signals[test_mask]], input_transform
            )
            validation_x, test_x = other
            validation_y = y[validation_mask]
        seed_probabilities: list[np.ndarray] = []
        for current_seed in seeds:
            model = _new_cnn(current, current_seed)
            if selection_protocol == "all_inner_refit":
                model.fit(train_x, y[train_mask], epochs=refit_epochs)
            else:
                model.fit(
                    train_x,
                    y[fit_mask],
                    validation_x=validation_x,
                    validation_y=validation_y,
                )
            seed_probabilities.append(
                model.predict_proba(test_x, int(current.get("inference_batch_size", 512)))
            )
        probabilities = np.mean(seed_probabilities, axis=0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        predicted = probabilities.argmax(axis=1)
        selected[str(fold_id)] = {
            "model_parameters": params,
            "selection_protocol": selection_protocol,
            "input_transform": input_transform,
            "seeds": list(seeds),
            "refit_epochs": refit_epochs,
            "channel_mean": channel_mean.ravel().tolist(),
            "channel_std": channel_std.ravel().tolist(),
        }
        for index, position in enumerate(np.flatnonzero(test_mask)):
            predictions.append({
                "outer_fold": fold_id,
                "subject_id": str(subjects[position]),
                "epoch_index": int(epoch_indices[position]),
                "y_true": int(y[position]),
                "y_pred": int(predicted[index]),
                **{f"prob_{STAGES[j]}": float(probabilities[index, j]) for j in range(4)},
            })
    return _write_result(predictions, selected, output_dir, "cnn", seed, int(config.get("bootstrap_repetitions", 2000)))


def _write_result(
    predictions: list[dict[str, Any]],
    selected: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    predictions.sort(key=lambda row: (int(row["outer_fold"]), str(row["subject_id"]), int(row["epoch_index"])))
    fields = list(predictions[0]) if predictions else []
    with (target / f"predictions_{model_name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    y_true = np.asarray([row["y_true"] for row in predictions], dtype=int)
    y_pred = np.asarray([row["y_pred"] for row in predictions], dtype=int)
    subjects = np.asarray([row["subject_id"] for row in predictions])
    metrics = _metrics(y_true, y_pred)
    metrics["confidence_intervals"] = _bootstrap_ci(y_true, y_pred, subjects, seed, repetitions)
    metrics["model"] = model_name
    metrics["n_subjects"] = int(np.unique(subjects).size)
    metrics["n_epochs"] = int(y_true.size)
    (target / f"metrics_{model_name}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (target / f"selected_parameters_{model_name}.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    return metrics
