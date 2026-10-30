from __future__ import annotations

import numpy as np


F1_NAMES = (
    "trimmed_mean", "median", "mad", "p05", "p10", "p25", "p75", "p90", "p95",
    "iqr", "p90_p10", "log1p_median", "log1p_iqr", "near_zero_fraction",
)
F2_NAMES = (
    "diff_median", "diff_mad", "diff_abs_p90", "second_diff_abs_median", "total_variation",
    "normalized_total_variation", "positive_slope_fraction", "negative_slope_fraction",
    "turning_points", "autocorrelation_lag1", "autocorrelation_lag2", "autocorrelation_lag3",
    "detrended_rms", "linear_slope", "quadratic_curvature", "first10_mean", "middle10_mean",
    "last10_mean", "last_first_mean_difference",
)
F4_NAMES = (
    "pearson_correlation", "spearman_correlation", "difference_correlation", "max_cross_correlation",
    "max_cross_correlation_lag", "median_difference", "median_absolute_difference",
    "median_normalized_difference", "iqr_difference", "iqr_absolute_difference",
    "iqr_normalized_difference", "p95_difference", "p95_absolute_difference",
    "p95_normalized_difference", "line_length_difference", "line_length_absolute_difference",
    "line_length_normalized_difference", "mean_sample_difference", "mad_sample_difference",
    "p95_absolute_sample_difference",
)


def _corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def _f1(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=1)
    trimmed = ordered[:, 3:-3].mean(axis=1)
    median = np.median(values, axis=1)
    p05, p10, p25, p75, p90, p95 = np.quantile(values, [0.05, 0.10, 0.25, 0.75, 0.90, 0.95], axis=1)
    iqr = p75 - p25
    return np.column_stack([
        trimmed, median, np.median(np.abs(values - median[:, None]), axis=1), p05, p10, p25,
        p75, p90, p95, iqr, p90 - p10, np.log1p(np.clip(median, 0, None)),
        np.log1p(np.clip(iqr, 0, None)), np.mean(np.abs(values) <= 1e-8, axis=1),
    ])


def _autocorrelation(values: np.ndarray, lag: int) -> np.ndarray:
    return _corr(values[:, :-lag], values[:, lag:])


def _f2(values: np.ndarray) -> np.ndarray:
    difference = np.diff(values, axis=1)
    second = np.diff(values, n=2, axis=1)
    median = np.median(difference, axis=1)
    total_variation = np.sum(np.abs(difference), axis=1)
    turning_points = np.sum(difference[:, 1:] * difference[:, :-1] < 0, axis=1)
    x = np.arange(values.shape[1], dtype=float)
    linear_design = np.column_stack([np.ones_like(x), x])
    linear_coefficients = values @ np.linalg.pinv(linear_design).T
    fitted = linear_coefficients @ linear_design.T
    quadratic_design = np.column_stack([np.ones_like(x), x, x * x])
    quadratic_coefficients = values @ np.linalg.pinv(quadratic_design).T
    first, middle, last = values[:, :10].mean(1), values[:, 10:20].mean(1), values[:, 20:].mean(1)
    return np.column_stack([
        median, np.median(np.abs(difference - median[:, None]), axis=1),
        np.quantile(np.abs(difference), 0.90, axis=1), np.median(np.abs(second), axis=1),
        total_variation, total_variation / max(values.shape[1] - 1, 1),
        np.mean(difference > 0, axis=1), np.mean(difference < 0, axis=1), turning_points,
        _autocorrelation(values, 1), _autocorrelation(values, 2), _autocorrelation(values, 3),
        np.sqrt(np.mean((values - fitted) ** 2, axis=1)), linear_coefficients[:, 1],
        quadratic_coefficients[:, 2], first, middle, last, last - first,
    ])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=float)[None, :]
    return ranks


def _normalised_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right) / (np.abs(left) + np.abs(right) + 1e-8)


def _f4(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_diff, right_diff = np.diff(left, axis=1), np.diff(right, axis=1)
    lags = range(-3, 4)
    correlations = []
    for lag in lags:
        if lag < 0:
            correlations.append(_corr(left[:, -lag:], right[:, :lag]))
        elif lag > 0:
            correlations.append(_corr(left[:, :-lag], right[:, lag:]))
        else:
            correlations.append(_corr(left, right))
    correlation_matrix = np.column_stack(correlations)
    best = np.argmax(correlation_matrix, axis=1)
    best_values = correlation_matrix[np.arange(left.shape[0]), best]
    best_lags = np.asarray(tuple(lags), dtype=float)[best]

    def stats(values: np.ndarray) -> tuple[np.ndarray, ...]:
        p25, p75, p95 = np.quantile(values, [0.25, 0.75, 0.95], axis=1)
        return np.median(values, axis=1), p75 - p25, p95, np.sum(np.abs(np.diff(values, axis=1)), axis=1)

    left_stats, right_stats = stats(left), stats(right)
    columns = [_corr(left, right), _corr(_ranks(left), _ranks(right)), _corr(left_diff, right_diff), best_values, best_lags]
    for l_value, r_value in zip(left_stats, right_stats):
        columns.extend([l_value - r_value, np.abs(l_value - r_value), _normalised_difference(l_value, r_value)])
    sample_difference = left - right
    sample_median = np.median(sample_difference, axis=1)
    columns.extend([
        sample_difference.mean(axis=1), np.median(np.abs(sample_difference - sample_median[:, None]), axis=1),
        np.quantile(np.abs(sample_difference), 0.95, axis=1),
    ])
    return np.column_stack(columns)


def extract_86_features(aeeg: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(aeeg, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (2, 30):
        raise ValueError(f"expected aEEG shape [N,2,30], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("aEEG values must be finite")
    matrices = [_f1(values[:, 0]), _f2(values[:, 0]), _f1(values[:, 1]), _f2(values[:, 1]), _f4(values[:, 0], values[:, 1])]
    names = [*(f"left_{name}" for name in F1_NAMES), *(f"left_{name}" for name in F2_NAMES),
             *(f"right_{name}" for name in F1_NAMES), *(f"right_{name}" for name in F2_NAMES),
             *(f"bilateral_{name}" for name in F4_NAMES)]
    output = np.column_stack(matrices).astype(np.float64, copy=False)
    if output.shape[1] != 86 or not np.isfinite(output).all():
        raise RuntimeError(f"expected 86 finite features, got {output.shape}")
    return output, names

