#!/usr/bin/env python3
"""Per-intervention regression diagnostics from a Prophet test_predictions.csv.

Standalone and dependency-light on purpose: only pandas/numpy/scipy/sklearn
are required (no torch, no `prophet` package), so this can run in a plain
venv on a laptop against any predictions CSV written by
train_standard_prophet.py -- most usefully test_predictions.csv, but it also
works on any dataframe with the same shape (cell line, intervention,
phenotype, y_true, y_pred).

This is deliberately separate from train_standard_prophet.py's own
compute_per_intervention_metrics(), which only reports interventions that
matched a skewness bin (anything missing from --gene-skewness-path /
--drug-skewness-path is silently dropped there). This script reports every
intervention present in the predictions file, with no skewness dependency.

Usage:
    python3 per_intervention_metrics.py \
        --predictions results/.../test_predictions.csv \
        --output-dir results/.../ \
        [--iv-col iv1] [--cell-line-col cell_line] [--phenotype-col phenotype] \
        [--group-by-phenotype / --no-group-by-phenotype] \
        [--min-cell-lines 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score


def regression_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression, calibration, dispersion, and lower-tail diagnostics.

    Copied verbatim from train_standard_prophet.py's _regression_diagnostics
    so this script has no import dependency on the trainer.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    n = len(y_true)
    target_std = float(np.std(y_true, ddof=1)) if n >= 2 else np.nan
    mse = float(mean_squared_error(y_true, y_pred)) if n else np.nan
    rmse = float(np.sqrt(mse)) if n else np.nan
    r2 = float(r2_score(y_true, y_pred)) if n >= 2 else np.nan
    spearman = spearmanr(y_true, y_pred).statistic if n >= 2 else np.nan
    pearson = pearsonr(y_true, y_pred).statistic if n >= 2 else np.nan
    calibration_slope = np.nan
    calibration_intercept = np.nan
    if n >= 2 and np.std(y_true) > 0:
        calibration_slope, calibration_intercept = np.polyfit(y_true, y_pred, 1)

    bottom_10_recall = np.nan
    bottom_10_average_precision = np.nan
    if n >= 10:
        true_tail = y_true <= np.quantile(y_true, 0.10)
        predicted_tail = y_pred <= np.quantile(y_pred, 0.10)
        if true_tail.any():
            bottom_10_recall = float((true_tail & predicted_tail).sum() / true_tail.sum())
            if np.unique(true_tail).size == 2:
                bottom_10_average_precision = float(
                    average_precision_score(true_tail.astype(int), -y_pred)
                )
    return {
        "n_observations": int(n),
        "target_mean": float(np.mean(y_true)) if n else np.nan,
        "target_std": target_std,
        "prediction_mean": float(np.mean(y_pred)) if n else np.nan,
        "prediction_std": float(np.std(y_pred, ddof=1)) if n >= 2 else np.nan,
        "r2": r2,
        "pearson": float(pearson) if not np.isnan(pearson) else np.nan,
        "spearman": float(spearman) if not np.isnan(spearman) else np.nan,
        "mse": mse,
        "rmse": rmse,
        "nrmse_target_std": (
            float(rmse / target_std)
            if np.isfinite(target_std) and target_std > 0
            else np.nan
        ),
        "mae": float(mean_absolute_error(y_true, y_pred)) if n else np.nan,
        "calibration_slope": float(calibration_slope),
        "calibration_intercept": float(calibration_intercept),
        "bottom_10_recall": bottom_10_recall,
        "bottom_10_average_precision": bottom_10_average_precision,
    }


def compute_per_intervention_metrics(
    predictions: pd.DataFrame,
    iv_col: str,
    cell_line_col: str,
    phenotype_col: str | None,
    group_by_phenotype: bool,
    min_cell_lines: int,
) -> pd.DataFrame:
    """One row of regression diagnostics per intervention.

    Multiple rows for the same (intervention, cell_line) pair -- e.g.
    replicate measurements -- are averaged first, so each cell line
    contributes one point to that intervention's R2/correlation rather than
    letting a heavily-replicated cell line dominate it.
    """
    required = {iv_col, cell_line_col, "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing)}")

    group_cols = [iv_col]
    if group_by_phenotype:
        if phenotype_col is None or phenotype_col not in predictions.columns:
            raise ValueError(
                "--group-by-phenotype requires --phenotype-col to name a "
                "column present in the predictions file"
            )
        group_cols = [phenotype_col, iv_col]

    per_cell_line = (
        predictions.groupby(group_cols + [cell_line_col], as_index=False, observed=True)[
            ["y_true", "y_pred"]
        ]
        .mean()
    )

    rows = []
    for keys, group in per_cell_line.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_cell_lines = int(group[cell_line_col].nunique())
        if n_cell_lines < min_cell_lines:
            continue
        row = dict(zip(group_cols, (str(key) for key in keys)))
        row["n_cell_lines"] = n_cell_lines
        row.update(
            regression_diagnostics(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
            )
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("r2", ascending=True, na_position="first").reset_index(
            drop=True
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="test_predictions.csv or similar")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iv-col", default="iv1")
    parser.add_argument("--cell-line-col", default="cell_line")
    parser.add_argument("--phenotype-col", default="phenotype")
    parser.add_argument(
        "--group-by-phenotype",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Group by (phenotype, intervention) rather than intervention alone. "
            "Keep this on if the same name could mean different things in "
            "different phenotypes (e.g. a gene knocked out in SCORE vs SCORE_ORG "
            "are on different scales); turn off to pool an intervention across "
            "every phenotype it appears in."
        ),
    )
    parser.add_argument(
        "--min-cell-lines",
        type=int,
        default=3,
        help="Drop interventions tested in fewer than this many cell lines (R2/correlation on 1-2 points is not meaningful).",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    metrics = compute_per_intervention_metrics(
        predictions,
        iv_col=args.iv_col,
        cell_line_col=args.cell_line_col,
        phenotype_col=args.phenotype_col,
        group_by_phenotype=args.group_by_phenotype,
        min_cell_lines=args.min_cell_lines,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "per_intervention_metrics.csv"
    metrics.to_csv(out_path, index=False)

    print(f"Wrote {len(metrics)} interventions to {out_path}")
    if not metrics.empty:
        print(
            metrics[["r2", "pearson", "spearman", "n_observations", "n_cell_lines"]]
            .describe()
            .to_string()
        )
        print("\nWorst 10 by R2:")
        print(metrics.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
