# prophet_mega (minimal)

Trains standard Prophet on drug-response and CRISPR-fitness data, then
reports per-intervention accuracy. Three files:

| File | Role |
|---|---|
| `train_standard_prophet.py` | Prepares data for Prophet's own model/dataset classes, drives training via PyTorch Lightning, writes predictions + summary reporting. |
| `per_intervention_metrics.py` | Standalone, run after training: turns `test_predictions.csv` into one row of R²/correlation per intervention. |
| `run_standard_prophet.sh` | Env-var-driven wrapper: assembles input paths, runs the two scripts above in sequence. |

## What Prophet itself already does

It's worth being clear about the division of labour, because most of the
actual modelling is not in this package at all — it's in the `prophet`
package (Theis lab), specifically two classes this package imports and
uses as-is, unmodified:

- **`PhenotypeDataset`** (`prophet.data.dataset`) — given a dataframe of
  rows plus an intervention-embedding table and a cell-line-embedding
  table, this is what actually turns each row into a model-ready item:
  looking up the right embedding vectors for the cell line and
  intervention(s), building the attention mask, and packaging the label.
  This package's job is only to hand `PhenotypeDataset` the right
  dataframe and the right two embedding tables — not to reimplement any
  of that lookup/packaging logic.
- **`TransformerPredictor`** (`prophet.models.transformer`) — Prophet's
  actual model: the transformer forward pass, the PyTorch Lightning
  `training_step`/`validation_step`/`test_step`/`predict_step`, and the
  loss computation. This is also where the weighted-MSE machinery
  actually lives: `training_step` reads a `response_weight` key off each
  training batch (if present) and applies it inside `_weighted_mse`;
  validation/test loss is always the *plain* MSE, regardless of
  weighting, so early stopping and reported test loss aren't affected by
  whatever training-time weighting scheme you're using. This package
  doesn't touch any of that — it only has to arrange for
  `response_weight` to be present on the right batches, which it does via
  `ResponseWeightedDataset` (see below).

So `train_standard_prophet.py` isn't a from-scratch trainer — it's
plumbing and bookkeeping *around* those two classes:

1. **Data assembly**: reads one or more Prophet-format CSVs
   (`--prophet-data`), restricts rows to cell lines present in your
   embedding table, optionally filters drugs/genes by skewness, optionally
   row-caps each split with a protected carve-out for named
   phenotypes/targets. None of this is Prophet's job — Prophet just
   receives whatever dataframe survives filtering.
2. **Embedding table assembly**: builds the intervention-embedding table
   (`build_standard_iv_embedding_table`) and validates the cell-line
   embedding table's shape (`--standard-cl-dim`/`--expected-standard-cl-dim`)
   before handing both to `PhenotypeDataset`.
3. **Weighted-MSE wiring**: `ResponseWeightedDataset` is a thin wrapper
   around whatever dataset it's given (here, `PhenotypeDataset`) that adds
   a precomputed `response_weight` value to each item — computed by
   `build_inverse_response_weights` and/or `build_skewness_response_weights`
   from this package. `TransformerPredictor` picks that key up on its own;
   this package never touches the loss.
4. **Splitting**: `grouped_split_indices` ensures no cell line appears in
   more than one of train/val/test — Prophet has no opinion on how you
   split, so this is entirely this package's responsibility.
5. **Orchestration**: builds a `pytorch_lightning.Trainer` (checkpointing,
   early stopping, CSV logging, two small custom callbacks for progress
   printing and per-epoch validation metrics) around the `TransformerPredictor`
   instance and calls `.fit()`/`.test()`/`.predict()` — standard Lightning
   usage, nothing Prophet-specific here either.
6. **Reporting beyond what Prophet computes**: Prophet's own
   `test_step`/`predict_step` give you predictions and an unweighted test
   MSE; everything past that — R²/Spearman/hit-ratio metrics, per-phenotype
   breakdowns, per-skewness-bin breakdowns — is computed by this package
   from the raw predictions, using `compute_hit_ratio` (the one utility
   function it does borrow from `prophet.utils.callbacks`).

In short: if you're trying to understand what actually predicts a value,
look in `prophet`. If you're trying to understand what data reached the
model and how, or what's reported afterwards, look here.

## Downstream: per-intervention metrics

Prophet's own test metrics are either pooled across every test row, or
(via this package's `compute_skewness_bin_metrics`) broken out by
skewness bin — but that breakdown only covers interventions that matched
a row in your `--gene-skewness-path`/`--drug-skewness-path` tables.
Anything missing from those tables is silently excluded from that report.

`per_intervention_metrics.py` is a separate, decoupled script that gives
every intervention in `test_predictions.csv` its own row of R², Pearson,
Spearman, MAE, calibration slope/intercept, and bottom-decile
recall/average-precision — no skewness table required. It:

- Averages replicate `(intervention, cell_line)` rows first, so one
  heavily-replicated cell line can't dominate an intervention's
  correlation.
- Groups by `(phenotype, intervention)` by default (`--group-by-phenotype`),
  since the same gene name means different things in `SCORE` vs
  `SCORE_ORG` — pass `--no-group-by-phenotype` to pool across phenotypes
  instead.
- Drops interventions tested in fewer than `--min-cell-lines` (default 3)
  cell lines, since R²/correlation on 1–2 points isn't meaningful.

It depends only on `pandas`/`numpy`/`scipy`/`sklearn` — no `torch`, no
`prophet` import — so it can run in a lighter environment than training
needs, on any machine, against any predictions CSV shaped like
`test_predictions.csv` (cell line, intervention, phenotype, `y_true`,
`y_pred` columns).

```bash
python3 per_intervention_metrics.py \
    --predictions results/.../test_predictions.csv \
    --output-dir results/.../
```

Writes `per_intervention_metrics.csv`, sorted worst-R² first, and prints a
quick summary + the 10 worst-performing interventions to stdout.

## End-to-end wrapper

`run_standard_prophet.sh` chains the two scripts together: assembles
`--prophet-data` from whichever `PROPHET_*_DATA` env vars are set, maps
the rest of the env vars onto `train_standard_prophet.py` flags, runs
training, then runs `per_intervention_metrics.py` against the resulting
`test_predictions.csv` automatically. See `SETUP.md` for the full env-var
reference and an example invocation.
