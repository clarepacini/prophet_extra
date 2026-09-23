#!/usr/bin/env python3
"""Train standard Prophet (no context-graph model) on GDSC/SCORE/organoid data.

Trimmed from train_graph_prophet.py: this file only supports the
--standard-prophet-only code path from the original trainer. There is no
graph model, no graph checkpoint, and no Omnipath/Sanger block-graph
pipeline -- the cell-line representation always comes from a fixed
embedding CSV (--standard-cl-embeddings).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate

try:
    from prophet.prophet.data.dataset import PhenotypeDataset
    from prophet.prophet.models.transformer import TransformerPredictor
    from prophet.prophet.utils.callbacks import compute_hit_ratio
except ImportError:
    from prophet.data.dataset import PhenotypeDataset
    from prophet.models.transformer import TransformerPredictor
    from prophet.utils.callbacks import compute_hit_ratio


def grouped_split_indices(
    cell_lines: np.ndarray,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_cell_lines = np.unique(cell_lines.astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_cell_lines)
    n_test = max(1, round(len(unique_cell_lines) * test_fraction))
    n_val = max(1, round(len(unique_cell_lines) * val_fraction))
    test_cls = set(unique_cell_lines[:n_test])
    val_cls = set(unique_cell_lines[n_test : n_test + n_val])
    test_idx = np.flatnonzero(np.isin(cell_lines, list(test_cls)))
    val_idx = np.flatnonzero(np.isin(cell_lines, list(val_cls)))
    train_idx = np.flatnonzero(
        ~np.isin(cell_lines, list(test_cls | val_cls))
    )
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Cell-line grouped split produced an empty partition")
    return train_idx, val_idx, test_idx


def stratified_subsample_indices(
    indices: np.ndarray,
    strata,
    max_rows: int,
    seed: int,
    sampling: str = "balanced",
    protected_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministically cap split rows while retaining every phenotype."""
    indices = np.asarray(indices, dtype=np.int64)
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    if protected_indices is not None:
        protected_indices = np.asarray(protected_indices, dtype=np.int64)
        protected = np.intersect1d(indices, protected_indices, assume_unique=False)
    else:
        protected = np.asarray([], dtype=np.int64)
    if len(protected) >= max_rows:
        return np.sort(protected)

    target_max_rows = max_rows - len(protected)
    candidate_indices = np.setdiff1d(indices, protected, assume_unique=False)
    if len(candidate_indices) <= target_max_rows:
        return np.sort(np.concatenate([protected, candidate_indices]))

    split_strata = np.asarray(strata)[candidate_indices].astype(str)
    labels, counts = np.unique(split_strata, return_counts=True)
    if target_max_rows < len(labels):
        raise ValueError(
            f"max_rows={max_rows} leaves only {target_max_rows} non-protected "
            f"slots, smaller than the {len(labels)} strata"
        )
    if sampling == "balanced":
        quotas = np.zeros(len(labels), dtype=np.int64)
        remaining = target_max_rows
        while remaining:
            candidates = np.flatnonzero(quotas < counts)
            if not len(candidates):
                break
            if remaining < len(candidates):
                quotas[candidates[:remaining]] += 1
                break
            increment = max(1, remaining // len(candidates))
            increments = np.minimum(counts[candidates] - quotas[candidates], increment)
            quotas[candidates] += increments
            remaining -= int(increments.sum())
    elif sampling == "proportional":
        quotas = np.maximum(
            1,
            np.floor(counts / counts.sum() * target_max_rows).astype(np.int64),
        )
        while quotas.sum() > target_max_rows:
            reducible = np.flatnonzero(quotas > 1)
            quotas[reducible[np.argmax(quotas[reducible])]] -= 1
        remainders = counts / counts.sum() * target_max_rows - quotas
        while quotas.sum() < target_max_rows:
            candidates = np.flatnonzero(quotas < counts)
            chosen = candidates[np.argmax(remainders[candidates])]
            quotas[chosen] += 1
            remainders[chosen] = -np.inf
    else:
        raise ValueError(f"Unknown phenotype sampling policy: {sampling}")

    rng = np.random.default_rng(seed)
    selected = []
    for label, quota in zip(labels, quotas):
        candidates = candidate_indices[split_strata == label]
        selected.extend(
            rng.choice(candidates, size=int(quota), replace=False).tolist()
        )
    return np.sort(np.concatenate([protected, np.asarray(selected, dtype=np.int64)]))


def protected_target_indices(
    data: pd.DataFrame,
    phenotype_col: str,
    iv_cols: Sequence[str],
    protected_phenotypes: Sequence[str],
    protected_targets: Sequence[str],
) -> np.ndarray:
    if not protected_phenotypes or not protected_targets:
        return np.asarray([], dtype=np.int64)
    phenotype_values = {str(value).lower() for value in protected_phenotypes}
    target_values = {str(value).lower() for value in protected_targets}
    mask = data[phenotype_col].astype(str).str.lower().isin(phenotype_values)
    target_mask = pd.Series(False, index=data.index)
    for iv_col in iv_cols:
        if iv_col in data.columns:
            target_mask |= data[iv_col].astype(str).str.lower().isin(target_values)
    return data.index[mask & target_mask].to_numpy(dtype=np.int64)


def value_counts_dict(values) -> Dict[str, int]:
    counts = pd.Series(values).astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def compute_prophet_style_test_metrics(
    prediction_rows: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    cell_line_col: str,
    phenotype_col: str,
    iv_cols: Sequence[str],
) -> Dict[str, float]:
    """Compute Prophet-style regression and cell-line/intervention ranking metrics."""
    y_pred = prediction_rows[prediction_col].to_numpy(dtype=np.float64)
    y_true = prediction_rows[target_col].to_numpy(dtype=np.float64)
    spearman = spearmanr(y_true, y_pred).statistic
    if np.isnan(spearman):
        spearman = 0.0

    interventions = [
        tuple(str(row[col]) for col in iv_cols if col in prediction_rows.columns)
        for _, row in prediction_rows.iterrows()
    ]
    hitratio_metrics = compute_hit_ratio(
        predictions=y_pred,
        targets=y_true,
        cl=prediction_rows[cell_line_col].astype(str).tolist(),
        phenotypes=prediction_rows[phenotype_col].astype(str).to_numpy(),
        iv=interventions,
        topk=[5, 10],
        suffix="_test",
    )
    metrics = {
        "R2_test": float(r2_score(y_true, y_pred)),
        "Spearman_test": float(spearman),
        **{key: float(value) for key, value in hitratio_metrics.items()},
        "test_loss": float(mean_squared_error(y_true, y_pred)),
    }
    return metrics


def _load_skewness_bins(
    path: Path,
    key_col: str,
    skew_col: str,
    n_bins: int,
) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [col for col in (key_col, skew_col) if col not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing skewness columns: {missing}")
    table = table[[key_col, skew_col]].dropna().copy()
    table[key_col] = table[key_col].astype(str)
    # Drug names can represent more than one GDSC drug ID. Prophet identifies
    # them by name, so collapse those entries to one mean signed skewness.
    table = table.groupby(key_col, as_index=False)[skew_col].mean()
    table["skewness_bin"] = (
        pd.qcut(table[skew_col], q=n_bins, labels=False, duplicates="drop") + 1
    ).astype("Int64")
    return table


def subset_score_genes_by_skewness_bin(
    rows: pd.DataFrame,
    phenotype_col: str,
    gene_col: str,
    skewness_path: Path,
    n_bins: int,
    genes_per_bin: int,
    min_absolute_skewness: float,
    seed: int,
    always_keep: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select genes globally by signed-skewness bin and retain all their rows."""
    if genes_per_bin < 0:
        raise ValueError("genes_per_bin must be non-negative")
    if min_absolute_skewness < 0:
        raise ValueError("min_absolute_skewness must be non-negative")
    if n_bins < 2:
        raise ValueError("--skewness-bins must be at least 2")
    if not skewness_path.exists():
        raise FileNotFoundError(f"Missing gene skewness file: {skewness_path}")

    score_mask = rows[phenotype_col].astype(str).str.contains(
        "SCORE", case=False, na=False
    )
    observed_genes = set(rows.loc[score_mask, gene_col].astype(str))
    bins = _load_skewness_bins(
        skewness_path,
        key_col="gene",
        skew_col="score2_fitness_skewness",
        n_bins=n_bins,
    )
    bins = bins[bins["gene"].isin(observed_genes)].copy()
    bins = bins[
        bins["score2_fitness_skewness"].abs() >= min_absolute_skewness
    ].copy()
    if bins.empty:
        raise ValueError("No SCORE genes passed mapping and absolute-skewness filtering")
    # The threshold is conceptually applied first: recompute quantile bins over
    # the eligible genes rather than retaining bins from the unfiltered table.
    bins["skewness_bin"] = (
        pd.qcut(
            bins["score2_fitness_skewness"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        + 1
    ).astype("Int64")
    protected = (
        {str(gene) for gene in always_keep}
        & observed_genes
        & set(bins["gene"].astype(str))
    )
    rng = np.random.default_rng(seed)
    selections = []
    for bin_number, group in bins.groupby("skewness_bin", sort=True):
        genes = np.sort(group["gene"].astype(str).unique())
        protected_in_bin = np.asarray(
            [gene for gene in genes if gene in protected], dtype=object
        )
        candidates = np.asarray(
            [gene for gene in genes if gene not in protected], dtype=object
        )
        target_count = len(genes) if genes_per_bin == 0 else genes_per_bin
        random_quota = max(0, target_count - len(protected_in_bin))
        chosen = rng.choice(
            candidates, size=min(random_quota, len(candidates)), replace=False
        )
        selections.extend(
            [
                pd.DataFrame({
                    "gene": chosen,
                    "skewness_bin": int(bin_number),
                    "selection_reason": "random_bin_sample",
                }),
                pd.DataFrame({
                    "gene": protected_in_bin,
                    "skewness_bin": int(bin_number),
                    "selection_reason": "protected_target",
                }),
            ]
        )
    selected = pd.concat(selections, ignore_index=True)

    already_selected = set(selected["gene"])
    extra_protected = sorted(protected - already_selected)
    if extra_protected:
        protected_bins = bins.set_index("gene")["skewness_bin"]
        selected = pd.concat(
            [
                selected,
                pd.DataFrame({
                    "gene": extra_protected,
                    "skewness_bin": [int(protected_bins.get(gene, -1)) for gene in extra_protected],
                    "selection_reason": "protected_target",
                }),
            ],
            ignore_index=True,
        )

    selected = selected.merge(
        bins[["gene", "score2_fitness_skewness"]], on="gene", how="left"
    ).sort_values(["skewness_bin", "gene"]).reset_index(drop=True)
    selected_genes = set(selected["gene"])
    keep_mask = ~score_mask | rows[gene_col].astype(str).isin(selected_genes)
    subset = rows.loc[keep_mask].reset_index(drop=True)
    selected_counts = (
        selected.groupby("skewness_bin", dropna=False)["gene"].nunique().to_dict()
    )
    summary = {
        "enabled": True,
        "genes_per_bin_requested": int(genes_per_bin),
        "minimum_absolute_skewness": float(min_absolute_skewness),
        "n_bins_requested": int(n_bins),
        "seed": int(seed),
        "n_score_genes_before": int(len(observed_genes)),
        "n_score_genes_selected": int(len(selected_genes)),
        "n_score_rows_before": int(score_mask.sum()),
        "n_score_rows_after": int(
            subset[phenotype_col].astype(str).str.contains("SCORE", case=False, na=False).sum()
        ),
        "selected_genes_per_bin": {str(k): int(v) for k, v in selected_counts.items()},
        "protected_targets_added": extra_protected,
    }
    return subset, selected, summary


def filter_drugs_by_skewness(
    rows: pd.DataFrame,
    phenotype_col: str,
    iv_col: str,
    drug_skewness_path: Path,
    maximum_skewness: float | None = None,
    minimum_absolute_skewness: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter drug rows using signed maximum and/or absolute minimum skewness."""
    if minimum_absolute_skewness < 0:
        raise ValueError("minimum_absolute_skewness must be non-negative")
    if not drug_skewness_path.exists():
        raise FileNotFoundError(f"Missing drug skewness file: {drug_skewness_path}")
    table = pd.read_csv(drug_skewness_path)
    required = {"DRUG_NAME", "ln_ic50_skewness"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{drug_skewness_path} is missing columns: {missing}")
    lookup = (
        table[["DRUG_NAME", "ln_ic50_skewness"]]
        .dropna()
        .assign(DRUG_NAME=lambda frame: frame["DRUG_NAME"].astype(str))
        .groupby("DRUG_NAME")["ln_ic50_skewness"]
        .mean()
    )

    is_gene = rows[phenotype_col].astype(str).str.contains(
        "SCORE", case=False, na=False
    )
    keys = rows[iv_col].astype(str).copy()
    if "iv_name" in rows.columns:
        named = rows["iv_name"].notna() & rows["iv_name"].astype(str).str.len().gt(0)
        keys.loc[~is_gene & named] = rows.loc[~is_gene & named, "iv_name"].astype(str)
    row_skewness = keys.map(lookup)
    mapped_drug = ~is_gene & row_skewness.notna()
    remove = mapped_drug & row_skewness.abs().lt(minimum_absolute_skewness)
    if maximum_skewness is not None:
        remove |= mapped_drug & row_skewness.gt(maximum_skewness)
    removed = pd.DataFrame({
        "drug": keys.loc[remove],
        "ln_ic50_skewness": row_skewness.loc[remove],
    }).drop_duplicates().sort_values(["ln_ic50_skewness", "drug"])
    filtered = rows.loc[~remove].reset_index(drop=True)
    summary = {
        "enabled": True,
        "minimum_absolute_drug_skewness": float(minimum_absolute_skewness),
        "maximum_signed_drug_skewness": (
            None if maximum_skewness is None else float(maximum_skewness)
        ),
        "comparison": (
            "retain abs(ln_ic50_skewness) >= minimum; optionally exclude "
            "ln_ic50_skewness > maximum"
        ),
        "n_drug_rows_before": int((~is_gene).sum()),
        "n_drug_rows_removed": int(remove.sum()),
        "n_drug_rows_after": int((~is_gene).sum() - remove.sum()),
        "n_drugs_removed": int(removed["drug"].nunique()),
        "n_unmapped_drug_rows_retained": int((~is_gene & row_skewness.isna()).sum()),
    }
    return filtered, removed, summary


def filter_drugs_by_maximum_skewness(
    rows: pd.DataFrame,
    phenotype_col: str,
    iv_col: str,
    drug_skewness_path: Path,
    maximum_skewness: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Backward-compatible wrapper for the original signed-maximum filter."""
    return filter_drugs_by_skewness(
        rows=rows,
        phenotype_col=phenotype_col,
        iv_col=iv_col,
        drug_skewness_path=drug_skewness_path,
        maximum_skewness=maximum_skewness,
    )


def build_inverse_response_weights(
    rows: pd.DataFrame,
    phenotype_col: str,
    readout_col: str,
    epsilon: float,
    max_weight: float,
) -> tuple[np.ndarray, dict]:
    """Weight SCORE errors inversely, normalized so response 1 has weight 1."""
    if epsilon <= 0:
        raise ValueError("--inverse-response-epsilon must be positive")
    if max_weight != 0 and max_weight < 1:
        raise ValueError("--inverse-response-max-weight must be 0 or at least 1")
    is_gene = rows[phenotype_col].astype(str).str.contains(
        "SCORE", case=False, na=False
    ).to_numpy()
    response = pd.to_numeric(rows[readout_col], errors="coerce").to_numpy(dtype=float)
    invalid = is_gene & (~np.isfinite(response) | (response < 0) | (response > 1))
    if invalid.any():
        examples = response[invalid][:5].tolist()
        raise ValueError(
            "Inverse response weighting expects SCORE values scaled to [0, 1]; "
            f"found {int(invalid.sum())} invalid rows, examples={examples}"
        )
    weights = np.ones(len(rows), dtype=np.float32)
    weights[is_gene] = (1.0 + epsilon) / (response[is_gene] + epsilon)
    if max_weight > 0:
        weights = np.minimum(weights, max_weight)
    summary = {
        "formula": "(1 + epsilon) / (scaled_response + epsilon), SCORE rows only",
        "epsilon": float(epsilon),
        "max_weight": float(max_weight),
        "n_score_rows": int(is_gene.sum()),
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
    }
    return weights, summary


class ResponseWeightedDataset(Dataset):
    """Add a precomputed response-loss weight without changing base datasets."""

    def __init__(self, dataset: Dataset, weights: np.ndarray):
        if len(dataset) != len(weights):
            raise ValueError("Response weights must align one-to-one with dataset rows")
        self.dataset = dataset
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["response_weight"] = self.weights[index]
        return item


def build_skewness_response_weights(
    rows: pd.DataFrame,
    train_indices: np.ndarray,
    phenotype_col: str,
    gene_skewness_path: Path,
    drug_skewness_path: Path,
    strength: float,
    max_weight: float,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Build 1 + strength * normalized absolute-skewness row weights."""
    if strength < 0:
        raise ValueError("--skewness-loss-strength must be non-negative")
    if max_weight != 0 and max_weight < 1:
        raise ValueError("--skewness-loss-max-weight must be 0 or at least 1")

    gene = _load_skewness_bins(
        gene_skewness_path, "gene", "score2_fitness_skewness", 10
    ).set_index("gene")["score2_fitness_skewness"].abs()
    drug = _load_skewness_bins(
        drug_skewness_path, "DRUG_NAME", "ln_ic50_skewness", 10
    ).set_index("DRUG_NAME")["ln_ic50_skewness"].abs()

    phenotype = rows[phenotype_col].astype(str)
    is_gene = phenotype.str.contains("SCORE", case=False, na=False)
    keys = rows["iv1"].astype(str).copy()
    if "iv_name" in rows.columns:
        named_drugs = rows["iv_name"].notna() & rows["iv_name"].astype(str).ne("")
        keys.loc[~is_gene & named_drugs] = rows.loc[~is_gene & named_drugs, "iv_name"].astype(str)

    absolute_skewness = pd.Series(np.nan, index=rows.index, dtype=float)
    absolute_skewness.loc[is_gene] = keys.loc[is_gene].map(gene)
    absolute_skewness.loc[~is_gene] = keys.loc[~is_gene].map(drug)
    abs_values = absolute_skewness.to_numpy(dtype=float)
    train_abs = abs_values[np.asarray(train_indices, dtype=int)]
    train_mean = float(np.nanmean(train_abs)) if np.isfinite(train_abs).any() else np.nan
    if not np.isfinite(train_mean) or train_mean <= 0:
        raise ValueError("No positive skewness values mapped onto the selected training rows")

    weights = np.ones(len(rows), dtype=np.float32)
    mapped = np.isfinite(abs_values)
    weights[mapped] = 1.0 + strength * (abs_values[mapped] / train_mean)
    if max_weight > 0:
        weights = np.minimum(weights, max_weight)

    diagnostics = pd.DataFrame({
        "intervention_type": np.where(is_gene, "gene", "drug"),
        "intervention": keys.to_numpy(),
        "absolute_skewness": abs_values,
        "response_weight": weights,
    })
    diagnostics = diagnostics.groupby(
        ["intervention_type", "intervention"], as_index=False, dropna=False
    ).agg(
        absolute_skewness=("absolute_skewness", "first"),
        response_weight=("response_weight", "first"),
        n_rows=("intervention", "size"),
    )
    summary = {
        "formula": "1 + strength * absolute_skewness / mean_train_absolute_skewness",
        "strength": float(strength),
        "max_weight": float(max_weight),
        "mean_train_absolute_skewness": train_mean,
        "n_rows": int(len(rows)),
        "n_rows_mapped": int(mapped.sum()),
        "n_rows_unmapped_weight_one": int((~mapped).sum()),
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
    }
    return weights, diagnostics, summary


def _regression_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression, calibration, dispersion, and lower-tail diagnostics."""
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
        "target_mean": float(np.mean(y_true)) if n else np.nan,
        "target_std": target_std,
        "target_q25": float(np.quantile(y_true, 0.25)) if n else np.nan,
        "target_median": float(np.median(y_true)) if n else np.nan,
        "target_q75": float(np.quantile(y_true, 0.75)) if n else np.nan,
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
    annotated: pd.DataFrame,
    split: str,
    cell_line_col: str,
) -> pd.DataFrame:
    """Calculate each gene/drug's performance across unique cell lines."""
    mapped = annotated.dropna(subset=["skewness_bin", "intervention_skewness"]).copy()
    observation_counts = mapped.groupby(
        ["intervention_type", "skewness_lookup_key"],
        observed=True,
    ).size()
    per_cell_line = (
        mapped.groupby(
            [
                "intervention_type",
                "skewness_lookup_key",
                "skewness_bin",
                "intervention_skewness",
                cell_line_col,
            ],
            as_index=False,
            observed=True,
        )[["y_true", "y_pred"]]
        .mean()
    )
    rows = []
    for keys, group in per_cell_line.groupby(
        [
            "intervention_type",
            "skewness_lookup_key",
            "skewness_bin",
            "intervention_skewness",
        ],
        observed=True,
    ):
        intervention_type, intervention, skewness_bin, skewness = keys
        row = {
            "split": split,
            "intervention_type": str(intervention_type),
            "intervention": str(intervention),
            "skewness_bin": int(skewness_bin),
            "intervention_skewness": float(skewness),
            "n_observations": int(
                observation_counts.loc[(intervention_type, intervention)]
            ),
            "n_cell_lines": int(group[cell_line_col].nunique()),
        }
        row.update(
            _regression_diagnostics(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_skewness_bin_metrics(
    predictions: pd.DataFrame,
    gene_skewness_path: Path,
    drug_skewness_path: Path,
    n_bins: int,
    split: str,
    phenotype_col: str,
    iv_col: str,
    cell_line_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach bins and summarize per-intervention, rather than pooled-row, metrics."""
    gene_bins = _load_skewness_bins(
        gene_skewness_path,
        key_col="gene",
        skew_col="score2_fitness_skewness",
        n_bins=n_bins,
    ).rename(
        columns={
            "gene": iv_col,
            "score2_fitness_skewness": "intervention_skewness",
        }
    )
    gene_bins["intervention_type"] = "gene"
    drug_bins = _load_skewness_bins(
        drug_skewness_path,
        key_col="DRUG_NAME",
        skew_col="ln_ic50_skewness",
        n_bins=n_bins,
    ).rename(
        columns={
            "DRUG_NAME": iv_col,
            "ln_ic50_skewness": "intervention_skewness",
        }
    )
    drug_bins["intervention_type"] = "drug"

    annotated = predictions.copy()
    annotated[iv_col] = annotated[iv_col].astype(str)
    gene_lookup = gene_bins.set_index(iv_col)
    drug_lookup = drug_bins.set_index(iv_col)
    phenotype_is_score = annotated[phenotype_col].astype(str).str.upper().str.contains(
        "SCORE"
    )
    annotated["intervention_type"] = np.where(
        phenotype_is_score, "gene", "drug"
    )
    annotated["skewness_lookup_key"] = annotated[iv_col].astype(str)
    # GDSC commonly stores a SMILES string in iv1 and the embedding-compatible
    # drug name in iv_name. Drug skewness is keyed by DRUG_NAME, whereas SCORE
    # gene skewness is keyed directly by iv1.
    if "iv_name" in annotated.columns:
        drug_rows = annotated["intervention_type"].eq("drug")
        drug_names = annotated.loc[drug_rows, "iv_name"]
        usable_names = drug_names.notna() & drug_names.astype(str).str.len().gt(0)
        annotated.loc[drug_names.index[usable_names], "skewness_lookup_key"] = (
            drug_names.loc[usable_names].astype(str)
        )
    annotated["intervention_skewness"] = np.nan
    annotated["skewness_bin"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    for intervention_type, lookup in (("gene", gene_lookup), ("drug", drug_lookup)):
        row_mask = annotated["intervention_type"].eq(intervention_type)
        keys = annotated.loc[row_mask, "skewness_lookup_key"]
        annotated.loc[row_mask, "intervention_skewness"] = keys.map(
            lookup["intervention_skewness"]
        ).to_numpy()
        annotated.loc[row_mask, "skewness_bin"] = keys.map(
            lookup["skewness_bin"]
        ).to_numpy()
    annotated["split"] = split

    mapped = annotated.dropna(subset=["skewness_bin", "intervention_skewness"])
    intervention_metrics = compute_per_intervention_metrics(
        annotated,
        split=split,
        cell_line_col=cell_line_col,
    )
    diagnostic_columns = [
        "target_mean",
        "target_std",
        "prediction_mean",
        "prediction_std",
        "r2",
        "pearson",
        "spearman",
        "mse",
        "rmse",
        "nrmse_target_std",
        "mae",
        "calibration_slope",
        "calibration_intercept",
        "bottom_10_recall",
        "bottom_10_average_precision",
    ]
    metric_rows = []
    bin_sources = {"gene": gene_bins, "drug": drug_bins}
    for intervention_type in ("gene", "drug"):
        source = bin_sources[intervention_type]
        for skewness_bin in range(1, n_bins + 1):
            group = mapped[
                mapped["intervention_type"].eq(intervention_type)
                & mapped["skewness_bin"].eq(skewness_bin)
            ]
            source_bin = source[source["skewness_bin"].eq(skewness_bin)]
            intervention_group = intervention_metrics[
                intervention_metrics["intervention_type"].eq(intervention_type)
                & intervention_metrics["skewness_bin"].eq(skewness_bin)
            ]
            row = {
                    "split": split,
                    "intervention_type": intervention_type,
                    "skewness_bin": skewness_bin,
                    "n_rows": int(len(group)),
                    "n_interventions": int(len(intervention_group)),
                    "bin_skewness_min": (
                        float(source_bin["intervention_skewness"].min())
                        if len(source_bin)
                        else np.nan
                    ),
                    "bin_skewness_max": (
                        float(source_bin["intervention_skewness"].max())
                        if len(source_bin)
                        else np.nan
                    ),
                    "observed_skewness_mean": (
                        float(intervention_group["intervention_skewness"].mean())
                        if len(intervention_group)
                        else np.nan
                    ),
            }
            # Pooled row-level performance for the complete bin.  These are
            # deliberately separate from the intervention-level summaries
            # below, which give every intervention equal weight regardless of
            # its number of observations.
            global_diagnostics = _regression_diagnostics(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
            )
            row["global_r2"] = global_diagnostics["r2"]
            row["global_pearson"] = global_diagnostics["pearson"]
            row["global_spearman"] = global_diagnostics["spearman"]
            for metric in diagnostic_columns:
                values = pd.to_numeric(
                    intervention_group.get(metric, pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna()
                prefix = f"intervention_{metric}"
                row[f"{prefix}_n"] = int(len(values))
                row[f"{prefix}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{prefix}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{prefix}_std"] = (
                    float(values.std(ddof=1)) if len(values) >= 2 else np.nan
                )
                row[f"{prefix}_q25"] = (
                    float(values.quantile(0.25)) if len(values) else np.nan
                )
                row[f"{prefix}_q75"] = (
                    float(values.quantile(0.75)) if len(values) else np.nan
                )
            metric_rows.append(row)

    bin_metrics = pd.DataFrame(metric_rows)
    return bin_metrics, intervention_metrics, annotated


def build_standard_iv_embedding_table(
    iv_embeddings: pd.DataFrame,
    intervention_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build Prophet IV embeddings keyed by names plus SMILES aliases."""
    iv_types = iv_embeddings[["type"]].groupby(
        iv_embeddings.index.astype(str),
        sort=False,
    ).first()
    table = pd.concat([iv_types, intervention_features], axis=1, join="inner")
    alias_tables = [table]

    if "smiles" in iv_embeddings.columns:
        smiles = iv_embeddings["smiles"].dropna().astype(str).str.lower()
        smiles = smiles[smiles.str.len() > 0]
        smiles = smiles[~smiles.index.duplicated(keep="first")]
        smiles_alias = table.reindex(smiles.index).dropna(subset=["type"]).copy()
        smiles_alias.index = smiles.loc[smiles_alias.index].values
        alias_tables.append(smiles_alias)

    feature_cols = list(intervention_features.columns)
    negative_rows = []
    for token, intervention_type in {
        "negative_gene": "gene",
        "negative_drug": "drug",
    }.items():
        if token not in table.index:
            negative_rows.append(
                pd.DataFrame(
                    [[intervention_type] + [0.0] * len(feature_cols)],
                    index=[token],
                    columns=["type"] + feature_cols,
                )
            )
    if negative_rows:
        alias_tables.extend(negative_rows)

    expanded = pd.concat(alias_tables, axis=0)
    expanded.index = expanded.index.astype(str)
    expanded = expanded[~expanded.index.duplicated(keep="first")]
    return expanded


def filter_rows_to_iv_embeddings(
    data: pd.DataFrame,
    iv_cols: Sequence[str],
    iv_embedding_index: pd.Index,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    valid_ids = set(iv_embedding_index.astype(str))
    valid = pd.Series(True, index=data.index)
    missing_counts = {}
    for iv_col in iv_cols:
        col_valid = data[iv_col].astype(str).isin(valid_ids)
        missing_counts[iv_col] = int((~col_valid).sum())
        valid &= col_valid
    filtered = data.loc[valid].copy()
    return filtered, {
        "n_rows_before": int(len(data)),
        "n_rows_after": int(len(filtered)),
        "n_rows_excluded": int((~valid).sum()),
        "missing_by_iv_col": missing_counts,
    }


def torch_gpu_memory_summary() -> Dict[str, object]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    max_allocated = int(torch.cuda.max_memory_allocated(device))
    max_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "cuda_available": True,
        "device_index": int(device),
        "device_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "total_memory_gb": float(props.total_memory / 1024**3),
        "current_allocated_bytes": allocated,
        "current_allocated_gb": float(allocated / 1024**3),
        "current_reserved_bytes": reserved,
        "current_reserved_gb": float(reserved / 1024**3),
        "max_allocated_bytes": max_allocated,
        "max_allocated_gb": float(max_allocated / 1024**3),
        "max_reserved_bytes": max_reserved,
        "max_reserved_gb": float(max_reserved / 1024**3),
    }


class PeriodicProgressCallback(pl.Callback):
    """Emit compact progress lines suitable for non-interactive farm logs."""

    def __init__(self, every_n_steps: int = 100):
        self.every_n_steps = max(1, int(every_n_steps))
        self.started_at = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self.started_at = time.monotonic()
        print(
            f"Training for at most {trainer.max_steps} steps; "
            f"progress every {self.every_n_steps} steps",
            flush=True,
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.every_n_steps:
            return
        metrics = trainer.callback_metrics
        loss = next(
            (
                metrics[key]
                for key in (
                    "train_loss_step",
                    "train_loss",
                    "training_loss",
                )
                if key in metrics
            ),
            None,
        )
        loss_text = (
            f"{float(loss.detach().cpu()):.6f}"
            if isinstance(loss, torch.Tensor)
            else "unavailable"
        )
        elapsed = time.monotonic() - self.started_at
        print(
            f"Progress step {step}/{trainer.max_steps}: "
            f"train_loss={loss_text}, elapsed={elapsed / 60:.1f} min",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or self.started_at is None:
            return
        loss = trainer.callback_metrics.get("validation_loss")
        if isinstance(loss, torch.Tensor):
            print(
                f"Validation at step {trainer.global_step}: "
                f"validation_loss={float(loss.detach().cpu()):.6f}",
                flush=True,
            )


class ProphetValidationMetricsCallback(pl.Callback):
    """Compute Prophet-style validation metrics on the retained split rows."""

    def __init__(
        self,
        retained_rows: pd.DataFrame,
        cell_line_col: str,
        phenotype_col: str,
        iv_cols: Sequence[str],
    ):
        self.retained_rows = retained_rows.reset_index(drop=True)
        self.cell_line_col = cell_line_col
        self.phenotype_col = phenotype_col
        self.iv_cols = list(iv_cols)
        self.predictions = []
        self.targets = []
        self.row_indices = []
        self.objective_weights = {
            "R2_validation": 0.45,
            "Spearman_validation": 0.25,
            "cl_avg_precision_both_10_validation": 0.15,
            "iv_avg_precision_both_10_validation": 0.15,
        }

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if outputs is None:
            return
        row_index = batch.get("row_index", batch.get("idx"))
        if row_index is None:
            return
        self.predictions.append(outputs["y_pred"].detach().float().cpu())
        self.targets.append(outputs["y_true"].detach().float().cpu())
        self.row_indices.append(row_index.detach().long().cpu())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not self.predictions:
            self._reset()
            return

        y_pred = torch.cat(self.predictions, dim=0).flatten().numpy()
        y_true = torch.cat(self.targets, dim=0).flatten().numpy()
        row_indices = torch.cat(self.row_indices, dim=0).flatten().numpy()
        rows = self.retained_rows.iloc[row_indices]

        r2 = float(r2_score(y_true, y_pred))
        spearman = spearmanr(y_true, y_pred).statistic
        if np.isnan(spearman):
            spearman = 0.0
        metrics = {
            "R2_validation": r2,
            "Spearman_validation": float(spearman),
        }
        interventions = [
            tuple(str(row[col]) for col in self.iv_cols if col in rows.columns)
            for _, row in rows.iterrows()
        ]
        hitratio_metrics = compute_hit_ratio(
            predictions=y_pred,
            targets=y_true,
            cl=rows[self.cell_line_col].astype(str).tolist(),
            phenotypes=rows[self.phenotype_col].astype(str).to_numpy(),
            iv=interventions,
            topk=[5, 10],
            suffix="_validation",
        )
        metrics.update({key: float(value) for key, value in hitratio_metrics.items()})

        objective = 0.0
        parts = []
        for metric, weight in self.objective_weights.items():
            if metric not in metrics:
                continue
            objective += metrics[metric] * weight
            parts.append(f"{metric}={metrics[metric]:.6g}*{weight:g}")
        metrics["validation_objective"] = float(objective)

        for name, value in metrics.items():
            pl_module.log(
                name,
                value,
                sync_dist=True,
                batch_size=int(y_true.shape[0]),
            )
        if trainer.is_global_zero:
            print(
                f"Validation objective validation_objective="
                f"{objective:.6g} ({', '.join(parts)})",
                flush=True,
            )
        self._reset()

    def _reset(self) -> None:
        self.predictions = []
        self.targets = []
        self.row_indices = []


def restrict_to_embedding_cell_lines(
    data: pd.DataFrame,
    cell_line_col: str,
    embedding_cell_lines,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Restrict response rows to the cell-line universe of an embedding table."""
    if cell_line_col not in data.columns:
        raise ValueError(f"Missing cell-line column: {cell_line_col}")
    embedding_ids = {str(cell_line) for cell_line in embedding_cell_lines}
    alias_to_embedding_id = {cell_line: cell_line for cell_line in embedding_ids}
    ambiguous_aliases = set()
    for cell_line in embedding_ids:
        if "__SIDM" not in cell_line:
            continue
        alias = cell_line.split("__SIDM", 1)[0]
        if alias in alias_to_embedding_id and alias_to_embedding_id[alias] != cell_line:
            ambiguous_aliases.add(alias)
            continue
        alias_to_embedding_id[alias] = cell_line
    for alias in ambiguous_aliases:
        if alias not in embedding_ids:
            alias_to_embedding_id.pop(alias, None)
    cell_lines = data[cell_line_col].astype(str)
    mapped_cell_lines = cell_lines.map(alias_to_embedding_id)
    keep = mapped_cell_lines.notna()
    excluded_counts = (
        data.loc[~keep, cell_line_col]
        .astype(str)
        .value_counts()
        .rename_axis(cell_line_col)
        .reset_index(name="n_rows")
    )
    filtered = data.loc[keep].copy()
    filtered[cell_line_col] = mapped_cell_lines.loc[keep].to_numpy()
    original_kept_cell_lines = cell_lines.loc[keep].to_numpy()
    filtered_cell_lines = filtered[cell_line_col].astype(str).to_numpy()
    n_rows_remapped = int(
        (original_kept_cell_lines != filtered_cell_lines).sum()
    )
    summary = {
        "n_rows_before": int(len(data)),
        "n_rows_after": int(len(filtered)),
        "n_rows_excluded": int((~keep).sum()),
        "n_rows_remapped_to_embedding_ids": n_rows_remapped,
        "n_cell_lines_before": int(cell_lines.nunique()),
        "n_cell_lines_after": int(
            filtered[cell_line_col].astype(str).nunique()
        ),
        "n_cell_lines_excluded": int(excluded_counts.shape[0]),
        "n_ambiguous_embedding_aliases": int(len(ambiguous_aliases)),
    }
    return filtered, summary, excluded_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prophet-data", type=Path, nargs="+", required=True)
    parser.add_argument("--iv-embeddings", type=Path, required=True)
    parser.add_argument(
        "--standard-cl-embeddings",
        type=Path,
        default=Path(
            "prophet_assets/embeddings/cell_line_embeddings/"
            "cell_line_embedding_full_ccle_300_scaled.csv"
        ),
    )
    parser.add_argument(
        "--standard-cl-dim",
        type=int,
        default=None,
        help=(
            "Number of standard Prophet cell-line embedding columns to expose "
            "to the tokenizer. Defaults to all columns in --standard-cl-embeddings."
        ),
    )
    parser.add_argument(
        "--expected-standard-cl-dim",
        type=int,
        default=None,
        help=(
            "Fail fast if --standard-cl-embeddings does not contain this many "
            "numeric columns. Useful for mutation-augmented runs."
        ),
    )
    parser.add_argument(
        "--restrict-to-standard-prophet-cell-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict every model to cell lines present in the standard "
            "Prophet cell-line embedding table"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iv-cols", nargs="+", default=["iv1"])
    parser.add_argument("--cell-line-col", default="cell_line")
    parser.add_argument("--phenotype-col", default="phenotype")
    parser.add_argument("--readout-col", default="value")
    parser.add_argument(
        "--gene-skewness-path",
        type=Path,
        default=Path(
            "prophet_assets/datasets/score2_fitness_skewness_by_gene.csv"
        ),
    )
    parser.add_argument(
        "--drug-skewness-path",
        type=Path,
        default=Path(
            "prophet_assets/datasets/gdsc2_ln_ic50_skewness_by_drug.csv"
        ),
    )
    parser.add_argument(
        "--maximum-drug-skewness",
        type=float,
        default=None,
        help="Exclude drugs with signed ln_IC50 skewness greater than this value.",
    )
    parser.add_argument(
        "--minimum-drug-absolute-skewness",
        type=float,
        default=0.0,
        help=(
            "Retain mapped drugs only when absolute ln_IC50 skewness is at least "
            "this value; unmapped drugs are retained."
        ),
    )
    parser.add_argument(
        "--skewness-bins",
        type=int,
        default=10,
        help="Quantile bins used for post-hoc gene/drug skewness metrics.",
    )
    parser.add_argument(
        "--genes-per-skewness-bin",
        type=int,
        default=0,
        help=(
            "Select this many SCORE genes from each signed-skewness bin and retain "
            "all available cell-line rows for them. Zero keeps all genes."
        ),
    )
    parser.add_argument(
        "--gene-subset-seed",
        type=int,
        default=2024,
        help="Random seed for reproducible gene selection within skewness bins.",
    )
    parser.add_argument(
        "--minimum-gene-absolute-skewness",
        type=float,
        default=0.0,
        help="Remove SCORE genes below this absolute-skewness threshold before bin sampling.",
    )
    parser.add_argument(
        "--skip-skewness-bin-metrics",
        action="store_true",
        help="Do not generate train/validation/test metrics by skewness bin.",
    )
    parser.add_argument(
        "--skewness-weighted-mse",
        action="store_true",
        help="Weight training-row MSE by the intervention's absolute skewness.",
    )
    parser.add_argument(
        "--skewness-loss-strength",
        type=float,
        default=1.0,
        help="Strength of absolute-skewness weighting (0 reproduces ordinary MSE).",
    )
    parser.add_argument(
        "--skewness-loss-max-weight",
        type=float,
        default=10.0,
        help="Cap skewness response weights; use 0 for no cap.",
    )
    parser.add_argument(
        "--inverse-response-weighted-mse",
        action="store_true",
        help="Weight SCORE gene-cell-line MSE by the inverse scaled response.",
    )
    parser.add_argument("--inverse-response-epsilon", type=float, default=0.05)
    parser.add_argument("--inverse-response-max-weight", type=float, default=10.0)
    parser.add_argument(
        "--combined-response-max-weight",
        type=float,
        default=10.0,
        help="Final cap after multiplying enabled response-weight components; 0 disables.",
    )
    parser.add_argument(
        "--target-col",
        default="Nominal Curated Therapuetic Target",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--gdsc-fraction", type=float, default=1.0)
    parser.add_argument("--auxiliary-fraction", type=float, default=1.0)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip the first (GDSC/drug) Prophet dataset and train only on auxiliary SCORE data.",
    )
    parser.add_argument("--gdsc-max-rows", type=int, default=0)
    parser.add_argument("--auxiliary-max-rows", type=int, default=0)
    parser.add_argument(
        "--phenotype-sampling",
        choices=["balanced", "proportional"],
        default="balanced",
        help=(
            "Allocation of capped rows among phenotypes. Balanced targets equal "
            "counts and fills shortages from phenotypes with available rows."
        ),
    )
    parser.add_argument(
        "--protected-subsample-phenotypes",
        nargs="*",
        default=["SCORE"],
        help=(
            "Phenotypes/datasets whose selected target rows are always kept "
            "when applying max row caps. Use an empty value from callers to disable."
        ),
    )
    parser.add_argument(
        "--protected-subsample-targets",
        nargs="*",
        default=["KRAS", "BRAF", "NRAS"],
        help=(
            "Intervention targets always kept for the protected phenotypes "
            "during max row caps."
        ),
    )
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument(
        "--evaluation-checkpoint",
        type=Path,
        default=None,
        help="Skip fitting and evaluate this trusted checkpoint",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=3000)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experimental_frames = []
    for dataset_index, path in enumerate(args.prophet_data):
        if args.score_only and dataset_index == 0:
            print(f"SCORE-only mode: skipping drug dataset {path}", flush=True)
            continue
        frame = pd.read_csv(path)
        fraction = args.gdsc_fraction if dataset_index == 0 else args.auxiliary_fraction
        max_rows = args.gdsc_max_rows if dataset_index == 0 else args.auxiliary_max_rows
        if dataset_index > 0 and (
            args.genes_per_skewness_bin > 0
            or args.minimum_gene_absolute_skewness > 0
        ):
            # Gene-level selection below needs the complete auxiliary cell-line
            # profiles; do not thin its rows before selecting genes.
            fraction = 1.0
            max_rows = 0
        if not 0 < fraction <= 1:
            raise ValueError(
                f"{'gdsc' if dataset_index == 0 else 'auxiliary'} fraction "
                f"must be in (0, 1], got {fraction}"
            )
        original_rows = len(frame)
        if fraction < 1:
            sampled = frame.sample(
                frac=fraction,
                replace=False,
                random_state=args.seed + dataset_index,
            )
            if dataset_index > 0 and args.iv_cols:
                protected_targets = {
                    str(value).lower() for value in args.protected_subsample_targets
                }
                protected = frame[
                    frame[args.iv_cols[0]].astype(str).str.lower().isin(protected_targets)
                ]
                sampled = pd.concat([sampled, protected], axis=0).drop_duplicates()
            frame = sampled
        if max_rows > 0 and len(frame) > max_rows:
            frame = frame.sample(
                n=max_rows,
                replace=False,
                random_state=args.seed + 100 + dataset_index,
            )
        frame = frame.reset_index(drop=True)
        print(
            f"Input sampling {path.name}: {original_rows:,} -> {len(frame):,} "
            f"(fraction={fraction}, max_rows={max_rows or None})",
            flush=True,
        )
        frame["_source_dataset"] = path.name
        frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
        experimental_frames.append(frame)
    if not experimental_frames:
        raise ValueError("No Prophet input datasets remain after dataset selection")
    experimental_data = pd.concat(experimental_frames, ignore_index=True)
    n_input_rows_before_cell_line_filter = int(len(experimental_data))
    input_phenotype_counts = value_counts_dict(
        experimental_data[args.phenotype_col]
    )
    standard_cl_embeddings = pd.read_csv(
        args.standard_cl_embeddings,
        index_col=0,
    ).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    standard_cl_embeddings.index = standard_cl_embeddings.index.astype(str)
    standard_cl_dim_available = int(standard_cl_embeddings.shape[1])
    if args.expected_standard_cl_dim is not None:
        if standard_cl_dim_available != args.expected_standard_cl_dim:
            raise ValueError(
                f"--standard-cl-embeddings has {standard_cl_dim_available} columns, "
                f"but --expected-standard-cl-dim={args.expected_standard_cl_dim}. "
                f"Path: {args.standard_cl_embeddings}"
            )
    standard_cl_dim = (
        int(args.standard_cl_dim)
        if args.standard_cl_dim is not None
        else standard_cl_dim_available
    )
    if standard_cl_dim <= 0 or standard_cl_dim > standard_cl_dim_available:
        raise ValueError(
            f"--standard-cl-dim must be between 1 and {standard_cl_dim_available}; "
            f"got {standard_cl_dim}."
        )
    standard_cl_embedding_summary = {
        "path": str(args.standard_cl_embeddings),
        "available_dim": standard_cl_dim_available,
        "model_dim_cl": standard_cl_dim,
        "first_columns": [str(col) for col in standard_cl_embeddings.columns[:10]],
        "first_ignored_columns": [
            str(col) for col in standard_cl_embeddings.columns[standard_cl_dim : standard_cl_dim + 10]
        ],
        "n_mutation_prefixed_columns": int(
            sum(str(col).lower().startswith("mutation_") for col in standard_cl_embeddings.columns)
        ),
        "n_score_pca_columns": int(
            sum(str(col).lower().startswith("score_pca_") for col in standard_cl_embeddings.columns)
        ),
        "n_ccle_prefixed_columns": int(
            sum(str(col).lower().startswith("ccle_") for col in standard_cl_embeddings.columns)
        ),
    }
    print("Standard Prophet cell-line embedding summary:", flush=True)
    print(json.dumps(standard_cl_embedding_summary, indent=2), flush=True)
    with (args.output_dir / "standard_cl_embedding_summary.json").open("w") as handle:
        json.dump(standard_cl_embedding_summary, handle, indent=2)
    restrict_cell_lines = args.restrict_to_standard_prophet_cell_lines
    if restrict_cell_lines:
        (
            experimental_data,
            cell_line_filter_summary,
            excluded_cell_lines,
        ) = restrict_to_embedding_cell_lines(
            experimental_data,
            args.cell_line_col,
            standard_cl_embeddings.index,
        )
        excluded_cell_lines.to_csv(
            args.output_dir / "excluded_non_prophet_cell_lines.csv",
            index=False,
        )
        with (args.output_dir / "prophet_cell_line_filter_summary.json").open(
            "w"
        ) as handle:
            json.dump(cell_line_filter_summary, handle, indent=2)
        if experimental_data.empty:
            raise ValueError(
                "No rows remain after restricting to standard Prophet "
                "cell-line embeddings"
            )
    else:
        cell_line_filter_summary = {
            "n_rows_before": int(len(experimental_data)),
            "n_rows_after": int(len(experimental_data)),
            "n_rows_excluded": 0,
            "n_cell_lines_before": int(
                experimental_data[args.cell_line_col].astype(str).nunique()
            ),
            "n_cell_lines_after": int(
                experimental_data[args.cell_line_col].astype(str).nunique()
            ),
            "n_cell_lines_excluded": 0,
        }
    post_cell_line_filter_phenotype_counts = value_counts_dict(
        experimental_data[args.phenotype_col]
    )
    drug_skewness_filter_summary = {"enabled": False}
    if (
        args.maximum_drug_skewness is not None
        or args.minimum_drug_absolute_skewness > 0
    ):
        experimental_data, excluded_drugs, drug_skewness_filter_summary = (
            filter_drugs_by_skewness(
                rows=experimental_data,
                phenotype_col=args.phenotype_col,
                iv_col=args.iv_cols[0],
                drug_skewness_path=args.drug_skewness_path,
                maximum_skewness=args.maximum_drug_skewness,
                minimum_absolute_skewness=args.minimum_drug_absolute_skewness,
            )
        )
        excluded_drugs.to_csv(
            args.output_dir / "excluded_drugs_by_skewness.csv", index=False
        )
        with (args.output_dir / "drug_skewness_filter_summary.json").open("w") as handle:
            json.dump(drug_skewness_filter_summary, handle, indent=2)
        print(f"Drug skewness filter: {drug_skewness_filter_summary}", flush=True)
    gene_subset_summary = {"enabled": False}
    if args.genes_per_skewness_bin > 0 or args.minimum_gene_absolute_skewness > 0:
        experimental_data, selected_genes, gene_subset_summary = (
            subset_score_genes_by_skewness_bin(
                rows=experimental_data,
                phenotype_col=args.phenotype_col,
                gene_col=args.iv_cols[0],
                skewness_path=args.gene_skewness_path,
                n_bins=args.skewness_bins,
                genes_per_bin=args.genes_per_skewness_bin,
                min_absolute_skewness=args.minimum_gene_absolute_skewness,
                seed=args.gene_subset_seed,
                always_keep=args.protected_subsample_targets,
            )
        )
        selected_genes.to_csv(
            args.output_dir / "selected_score_genes_by_skewness_bin.csv", index=False
        )
        with (args.output_dir / "score_gene_subset_summary.json").open("w") as handle:
            json.dump(gene_subset_summary, handle, indent=2)
        print(f"SCORE gene-level subset: {gene_subset_summary}", flush=True)
    iv_embeddings = pd.read_csv(args.iv_embeddings, index_col=0)
    iv_embeddings.index = iv_embeddings.index.astype(str)
    standard_intervention_features = iv_embeddings.drop(
        columns=["type", "smiles"],
        errors="ignore",
    ).apply(pd.to_numeric, errors="coerce")
    standard_intervention_features = standard_intervention_features.dropna(
        axis=1,
        how="all",
    ).fillna(0.0)
    standard_intervention_features = standard_intervention_features.groupby(
        standard_intervention_features.index.astype(str),
        sort=False,
    ).first()
    # Kept as an empty mapping so end-of-run provenance can report zero mapped
    # interventions without a graph-only variable disappearing from summary.json.
    intervention_to_gene = {}
    # Standard Prophet has no graph target-node input. Start directly from the
    # cell-line-filtered experimental rows so nominal drug-target annotations
    # cannot determine which drugs are retained.
    retained = experimental_data.copy()
    expected_iv_cols = [
        f"iv{position}" for position in range(1, len(args.iv_cols) + 1)
    ]
    if args.iv_cols != expected_iv_cols:
        raise ValueError(
            "Standard Prophet requires intervention columns named "
            f"{expected_iv_cols}; received {args.iv_cols}"
        )
    standard_iv_embeddings = build_standard_iv_embedding_table(
        iv_embeddings,
        standard_intervention_features,
    )
    retained, standard_iv_filter_summary = filter_rows_to_iv_embeddings(
        retained,
        args.iv_cols,
        standard_iv_embeddings.index,
    )
    if retained.empty:
        raise ValueError(
            "No standard Prophet rows remain after filtering to "
            "available intervention embeddings"
        )
    retained_phenotype_counts = value_counts_dict(retained[args.phenotype_col])
    phenotypes = sorted(retained[args.phenotype_col].astype(str).unique())
    dataset = PhenotypeDataset(
        experimental_data=retained,
        label_key=args.readout_col,
        iv_embeddings=standard_iv_embeddings,
        cell_line_embeddings=standard_cl_embeddings,
        phenotypes=phenotypes,
        pert_len=len(args.iv_cols),
    )
    retained_phenotype_counts = value_counts_dict(retained[args.phenotype_col])
    retained.to_csv(args.output_dir / "retained_training_rows.csv", index=False)
    train_idx, val_idx, test_idx = grouped_split_indices(
        retained[args.cell_line_col].astype(str).to_numpy(),
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    phenotype_strata = retained[args.phenotype_col].astype(str).to_numpy()
    protected_idx = protected_target_indices(
        retained,
        phenotype_col=args.phenotype_col,
        iv_cols=args.iv_cols,
        protected_phenotypes=args.protected_subsample_phenotypes,
        protected_targets=args.protected_subsample_targets,
    )
    if args.genes_per_skewness_bin > 0 or args.minimum_gene_absolute_skewness > 0:
        all_score_idx = np.flatnonzero(
            retained[args.phenotype_col].astype(str).str.contains(
                "SCORE", case=False, na=False
            ).to_numpy()
        )
        protected_idx = np.union1d(protected_idx, all_score_idx)
    protected_subsample_summary = {
        "phenotypes": list(args.protected_subsample_phenotypes),
        "targets": list(args.protected_subsample_targets),
        "n_rows_total": int(len(protected_idx)),
        "n_train_rows_before_cap": int(len(np.intersect1d(train_idx, protected_idx))),
        "n_val_rows_before_cap": int(len(np.intersect1d(val_idx, protected_idx))),
        "n_test_rows_before_cap": int(len(np.intersect1d(test_idx, protected_idx))),
    }
    print("Protected subsample rows:", flush=True)
    print(json.dumps(protected_subsample_summary, indent=2), flush=True)
    train_cap = args.max_train_rows
    val_cap = args.max_val_rows
    test_cap = args.max_test_rows
    if args.genes_per_skewness_bin > 0 or args.minimum_gene_absolute_skewness > 0:
        # The row caps apply to the remaining drug rows; complete SCORE gene
        # profiles are added without being independently row-sampled.
        train_cap = train_cap + len(np.intersect1d(train_idx, protected_idx)) if train_cap > 0 else 0
        val_cap = val_cap + len(np.intersect1d(val_idx, protected_idx)) if val_cap > 0 else 0
        test_cap = test_cap + len(np.intersect1d(test_idx, protected_idx)) if test_cap > 0 else 0
    train_idx = stratified_subsample_indices(
        train_idx,
        phenotype_strata,
        train_cap,
        args.seed + 11,
        args.phenotype_sampling,
        protected_indices=protected_idx,
    )
    val_idx = stratified_subsample_indices(
        val_idx,
        phenotype_strata,
        val_cap,
        args.seed + 12,
        args.phenotype_sampling,
        protected_indices=protected_idx,
    )
    test_idx = stratified_subsample_indices(
        test_idx,
        phenotype_strata,
        test_cap,
        args.seed + 13,
        args.phenotype_sampling,
        protected_indices=protected_idx,
    )
    protected_subsample_summary.update(
        {
            "n_train_rows_after_cap": int(len(np.intersect1d(train_idx, protected_idx))),
            "n_val_rows_after_cap": int(len(np.intersect1d(val_idx, protected_idx))),
            "n_test_rows_after_cap": int(len(np.intersect1d(test_idx, protected_idx))),
        }
    )
    with (args.output_dir / "protected_subsample_summary.json").open("w") as handle:
        json.dump(protected_subsample_summary, handle, indent=2)
    print(
        "Split rows after optional caps: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
        flush=True,
    )
    split_phenotype_counts = {
        "train": value_counts_dict(retained.iloc[train_idx][args.phenotype_col]),
        "val": value_counts_dict(retained.iloc[val_idx][args.phenotype_col]),
        "test": value_counts_dict(retained.iloc[test_idx][args.phenotype_col]),
    }
    print(
        "Phenotype rows after mapping: "
        f"{retained_phenotype_counts}; split counts: {split_phenotype_counts}",
        flush=True,
    )

    loader_dataset = dataset
    response_weights = np.ones(len(retained), dtype=np.float32)
    response_weight_summary = {"components": []}
    if args.skewness_weighted_mse:
        missing_paths = [
            str(path)
            for path in (args.gene_skewness_path, args.drug_skewness_path)
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Skewness-weighted MSE requires these files: {missing_paths}"
            )
        skewness_weights, weight_table, weight_summary = build_skewness_response_weights(
            rows=retained,
            train_indices=train_idx,
            phenotype_col=args.phenotype_col,
            gene_skewness_path=args.gene_skewness_path,
            drug_skewness_path=args.drug_skewness_path,
            strength=args.skewness_loss_strength,
            max_weight=args.skewness_loss_max_weight,
        )
        response_weights *= skewness_weights
        response_weight_summary["components"].append(
            {"name": "absolute_skewness", **weight_summary}
        )
        weight_table.to_csv(
            args.output_dir / "skewness_training_weights_by_intervention.csv", index=False
        )
        with (args.output_dir / "skewness_training_weight_summary.json").open("w") as handle:
            json.dump(weight_summary, handle, indent=2)
        print(f"Skewness-weighted training MSE: {weight_summary}", flush=True)
    if args.inverse_response_weighted_mse:
        inverse_weights, inverse_summary = build_inverse_response_weights(
            rows=retained,
            phenotype_col=args.phenotype_col,
            readout_col=args.readout_col,
            epsilon=args.inverse_response_epsilon,
            max_weight=args.inverse_response_max_weight,
        )
        response_weights *= inverse_weights
        response_weight_summary["components"].append(
            {"name": "inverse_response", **inverse_summary}
        )
        print(f"Inverse-response-weighted training MSE: {inverse_summary}", flush=True)
    if response_weight_summary["components"]:
        if args.combined_response_max_weight != 0 and args.combined_response_max_weight < 1:
            raise ValueError("--combined-response-max-weight must be 0 or at least 1")
        if args.combined_response_max_weight > 0:
            response_weights = np.minimum(
                response_weights, args.combined_response_max_weight
            )
        response_weight_summary.update({
            "combined_max_weight": float(args.combined_response_max_weight),
            "combined_weight_min": float(response_weights.min()),
            "combined_weight_mean": float(response_weights.mean()),
            "combined_weight_max": float(response_weights.max()),
        })
        loader_dataset = ResponseWeightedDataset(dataset, response_weights)
        pd.DataFrame({
            "row_index": np.arange(len(retained)),
            "phenotype": retained[args.phenotype_col].astype(str).to_numpy(),
            "intervention": retained[args.iv_cols[0]].astype(str).to_numpy(),
            "cell_line": retained[args.cell_line_col].astype(str).to_numpy(),
            "scaled_response": pd.to_numeric(
                retained[args.readout_col], errors="coerce"
            ).to_numpy(),
            "combined_response_weight": response_weights,
        }).to_csv(args.output_dir / "training_response_weights_by_row.csv", index=False)
        with (args.output_dir / "training_response_weight_summary.json").open("w") as handle:
            json.dump(response_weight_summary, handle, indent=2)

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(os.environ.get("PROPHET_NUM_WORKERS", "0"))
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(loader_dataset, train_idx),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(loader_dataset, val_idx),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(loader_dataset, test_idx),
        shuffle=False,
        **loader_kwargs,
    )

    common_model_args = dict(
        dim_phe=args.model_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        iv_dropout=0.1,
        cl_dropout=0.1,
        ph_dropout=0.1,
        regressor_dropout=0.1,
        lr=args.learning_rate,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        max_iters=args.max_steps,
        batch_size=args.batch_size,
        dropout=0.1,
        pool="cls",
        simpler=True,
        ctx_len=len(args.iv_cols) + 1,
        mask=True,
        sum=False,
        explicit_phenotype=False,
        linear_predictor=False,
        tokenizer_layers=2,
        seed=args.seed,
    )
    model = TransformerPredictor(
        dim_cl=standard_cl_dim,
        dim_iv=int(standard_intervention_features.shape[1]),
        **common_model_args,
    )
    model_type = "standard_prophet"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        monitor="validation_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        filename=f"{model_type}-{{epoch:02d}}-{{validation_loss:.5f}}",
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_steps=args.max_steps,
        callbacks=[
            checkpoint_callback,
            ProphetValidationMetricsCallback(
                retained_rows=retained,
                cell_line_col=args.cell_line_col,
                phenotype_col=args.phenotype_col,
                iv_cols=args.iv_cols,
            ),
            PeriodicProgressCallback(args.progress_interval),
            EarlyStopping(
                monitor="validation_loss",
                mode="min",
                patience=args.patience,
            ),
        ],
        logger=CSVLogger(
            save_dir=str(args.output_dir),
            name="lightning_logs",
        ),
        log_every_n_steps=10,
        enable_progress_bar=False,
        val_check_interval=max(
            1,
            min(
                args.val_check_interval,
                max(1, args.max_steps // 2),
            ),
        ),
        check_val_every_n_epoch=None,
    )
    if args.evaluation_checkpoint is None:
        trainer.fit(model, train_loader, val_loader)
        checkpoint_for_evaluation = "best"
        best_model_path = checkpoint_callback.best_model_path
    else:
        if not args.evaluation_checkpoint.is_file():
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: {args.evaluation_checkpoint}"
            )
        checkpoint_for_evaluation = str(args.evaluation_checkpoint)
        best_model_path = str(args.evaluation_checkpoint)
    test_metrics = trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path=checkpoint_for_evaluation,
        weights_only=False,
    )
    prediction_batches = trainer.predict(
        model,
        dataloaders=test_loader,
        ckpt_path=checkpoint_for_evaluation,
        weights_only=False,
    )
    y_pred = np.concatenate(
        [batch[0].detach().cpu().numpy().reshape(-1) for batch in prediction_batches]
    )
    y_true = np.concatenate(
        [batch[1].detach().cpu().numpy().reshape(-1) for batch in prediction_batches]
    )
    prediction_rows = retained.iloc[test_idx].copy().reset_index(drop=True)
    prediction_rows["y_true"] = y_true
    prediction_rows["y_pred"] = y_pred
    prediction_rows.to_csv(
        args.output_dir / "test_predictions.csv",
        index=False,
    )

    skewness_bin_metrics = []
    skewness_intervention_metrics = []
    if not args.skip_skewness_bin_metrics:
        if args.skewness_bins < 2:
            raise ValueError("--skewness-bins must be at least 2")
        missing_skewness_paths = [
            path
            for path in (args.gene_skewness_path, args.drug_skewness_path)
            if not path.is_file()
        ]
        if missing_skewness_paths:
            print(
                "Skipping skewness-bin metrics because files are missing: "
                f"{[str(path) for path in missing_skewness_paths]}",
                flush=True,
            )
        else:
            split_specs = [
                ("train", train_idx, None),
                ("validation", val_idx, None),
                ("test", test_idx, prediction_rows),
            ]
            coverage_rows = []
            for split_name, split_indices, existing_predictions in split_specs:
                if existing_predictions is None:
                    evaluation_loader = DataLoader(
                        Subset(dataset, split_indices),
                        shuffle=False,
                        **loader_kwargs,
                    )
                    split_batches = trainer.predict(
                        model,
                        dataloaders=evaluation_loader,
                        ckpt_path=checkpoint_for_evaluation,
                        weights_only=False,
                    )
                    split_y_pred = np.concatenate(
                        [
                            batch[0].detach().cpu().numpy().reshape(-1)
                            for batch in split_batches
                        ]
                    )
                    split_y_true = np.concatenate(
                        [
                            batch[1].detach().cpu().numpy().reshape(-1)
                            for batch in split_batches
                        ]
                    )
                    split_predictions = (
                        retained.iloc[split_indices].copy().reset_index(drop=True)
                    )
                    split_predictions["y_true"] = split_y_true
                    split_predictions["y_pred"] = split_y_pred
                else:
                    split_predictions = existing_predictions

                split_metrics, intervention_metrics, annotated = compute_skewness_bin_metrics(
                    predictions=split_predictions,
                    gene_skewness_path=args.gene_skewness_path,
                    drug_skewness_path=args.drug_skewness_path,
                    n_bins=args.skewness_bins,
                    split=split_name,
                    phenotype_col=args.phenotype_col,
                    iv_col=args.iv_cols[0],
                    cell_line_col=args.cell_line_col,
                )
                skewness_bin_metrics.append(split_metrics)
                skewness_intervention_metrics.append(intervention_metrics)
                coverage_rows.append(
                    {
                        "split": split_name,
                        "n_rows": int(len(annotated)),
                        "n_rows_with_skewness_bin": int(
                            annotated["skewness_bin"].notna().sum()
                        ),
                        "n_rows_without_skewness_bin": int(
                            annotated["skewness_bin"].isna().sum()
                        ),
                    }
                )
            skewness_bin_metrics = pd.concat(
                skewness_bin_metrics,
                ignore_index=True,
            )
            skewness_bin_metrics.to_csv(
                args.output_dir / "skewness_bin_metrics_by_split.csv",
                index=False,
            )
            pd.concat(
                skewness_intervention_metrics,
                ignore_index=True,
            ).to_csv(
                args.output_dir / "skewness_metrics_per_intervention_by_split.csv",
                index=False,
            )
            pd.DataFrame(coverage_rows).to_csv(
                args.output_dir / "skewness_bin_coverage_by_split.csv",
                index=False,
            )
            print(
                "Wrote post-hoc skewness-bin metrics for train, validation, and test",
                flush=True,
            )

    overall_metrics = {
        "n_rows": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }
    prophet_style_metrics = compute_prophet_style_test_metrics(
        prediction_rows=prediction_rows,
        prediction_col="y_pred",
        target_col="y_true",
        cell_line_col=args.cell_line_col,
        phenotype_col=args.phenotype_col,
        iv_cols=args.iv_cols,
    )
    pd.DataFrame(
        [
            {"metric": metric, "value": value}
            for metric, value in prophet_style_metrics.items()
        ]
    ).to_csv(
        args.output_dir / "prophet_style_test_metrics.csv",
        index=False,
    )
    print("Prophet-style test metrics:", flush=True)
    print(json.dumps(prophet_style_metrics, indent=2), flush=True)

    phenotype_metrics = []
    for phenotype, group in prediction_rows.groupby(args.phenotype_col):
        if len(group) < 2:
            continue
        phenotype_metrics.append(
            {
                "phenotype": str(phenotype),
                "n_rows": int(len(group)),
                "r2": float(r2_score(group["y_true"], group["y_pred"])),
                "spearman": float(
                    spearmanr(group["y_true"], group["y_pred"]).statistic
                ),
                "mse": float(
                    mean_squared_error(group["y_true"], group["y_pred"])
                ),
            }
        )
    pd.DataFrame(phenotype_metrics).to_csv(
        args.output_dir / "test_metrics_by_phenotype.csv",
        index=False,
    )
    gpu_memory = torch_gpu_memory_summary()
    with (args.output_dir / "gpu_memory_summary.json").open("w") as handle:
        json.dump(gpu_memory, handle, indent=2)
    print(f"GPU memory summary: {gpu_memory}", flush=True)

    summary = {
        "model_type": model_type,
        "n_input_rows_before_cell_line_filter": (
            n_input_rows_before_cell_line_filter
        ),
        "n_input_rows": int(len(experimental_data)),
        "input_phenotype_counts": input_phenotype_counts,
        "post_cell_line_filter_phenotype_counts": (
            post_cell_line_filter_phenotype_counts
        ),
        "retained_phenotype_counts": retained_phenotype_counts,
        "phenotype_sampling": args.phenotype_sampling,
        "protected_subsample": protected_subsample_summary,
        "split_phenotype_counts": split_phenotype_counts,
        "standard_iv_filter": standard_iv_filter_summary,
        "standard_cl_embedding": standard_cl_embedding_summary,
        "prophet_cell_line_filter": cell_line_filter_summary,
        "n_retained_rows": int(len(dataset)),
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_mapped_interventions": int(len(intervention_to_gene)),
        "graph_initialization": "not_used_standard_baseline",
        "active_genes_per_sample": None,
        "intervention_token_mode": "standard_prophet_embeddings",
        "best_checkpoint": best_model_path,
        "test_metrics": test_metrics,
        "prediction_metrics": overall_metrics,
        "prophet_style_test_metrics": prophet_style_metrics,
        "prediction_metrics_by_phenotype": phenotype_metrics,
        "gpu_memory": gpu_memory,
    }
    with open(args.output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
