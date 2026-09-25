# Setup

This package has three files (see `README.md` for what each does):
`train_standard_prophet.py`, `per_intervention_metrics.py`,
`run_standard_prophet.sh`. It depends on the `prophet` package (Theis
lab) for training, but `per_intervention_metrics.py` can run in a much
lighter environment on its own — see §4. There is data available in this google drive folder:
https://drive.google.com/drive/folders/1GcWsosTeWTZXuur3RFjI9MZO2vRooqi2?usp=sharing

## 1. Clone and install `prophet`

`prophet` is not included here — it's a separate checkout that must sit
next to this folder (or anywhere importable from your Python environment):

```
new_project/
├── prophet/           <- clone this
└── prophet_extra/       <- this package
```

```bash
cd new_project
git clone https://github.com/theislab/prophet.git
python3 -m venv prophet_env
source prophet_env/bin/activate
pip install -e ./prophet
```

## 2. Install the remaining Python dependencies

Into the same `prophet_env`:

```bash
pip install torch pytorch-lightning pandas numpy scipy scikit-learn
```

(Use whatever `torch`/CUDA build matches your GPU node, per your cluster's
usual instructions — this repo doesn't pin a specific version.)

## 3. Required data assets

None of these are included in this package. You need:

| Argument | What it is |
|---|---|
| `--prophet-data` (one or more) | Long-format Prophet CSVs: `row_id, cell_line, value, iv1, [iv2...], phenotype`. Typically GDSC2 drug response + SCORE CRISPR + organoid variants. |
| `--iv-embeddings` | Intervention (gene/drug) embedding CSV, indexed by intervention name, with a `type` column (`gene`/`drug`) and optionally `smiles`. This is what `PhenotypeDataset` looks each intervention up in — this package only builds the table, Prophet does the lookup. |
| `--standard-cl-embeddings` | Cell-line embedding CSV, indexed by cell-line ID. Same relationship: this package validates/passes it through, `PhenotypeDataset` does the actual per-row lookup. |
| `--gene-skewness-path` / `--drug-skewness-path` | Only needed if using `--genes-per-skewness-bin`, `--minimum-gene-absolute-skewness`, `--maximum-drug-skewness`, `--minimum-drug-absolute-skewness`, or the post-hoc skewness-bin metrics (on by default; pass `--skip-skewness-bin-metrics` to turn them off). |

You are able to use your own embeddings by changing the --iv-embeddings for interventions and --standard-cl-embeddings for model embeddings. 

## 4. Per-intervention metrics: lighter environment option

`per_intervention_metrics.py` imports only `pandas`, `numpy`, `scipy`, and
`sklearn` — no `torch`, no `prophet`. If you want to hand this analysis
step to a collaborator without the full training setup, they only need:

```bash
pip install pandas numpy scipy scikit-learn
```

and a copy of whatever `test_predictions.csv` your training run produced. The script generated the per-intervention, as opposed to global intervention statistics from the prophet run. These are useful for assessing model performance and more informative than global statistics alone.

## 5. Example run: the two scripts directly

This command shows is an example using a custom, not definitive, embedding for the models. The embedding has  expression data using cell lines and contains mutation embeddings. The default size is 512 but if your embedding is a different size, pass this with -standard-cl-dim. This example is using a skewness filter for including genes from the SCORE data (minimum-gene-absolute-skewness) and genes-per-skewness-bin. It is also using weights for observations during training, through the three inverse parameters. 

```bash
source prophet_env/bin/activate

python3 train_standard_prophet.py \
  --prophet-data prophet_extra_inputs/GDSC2_dataset_prophet.csv \
                 prophet_extra_inputs/SCORE2_dataset.csv \
  --iv-embeddings prophet_extra_inputs/iv_embeddings.csv \
  --standard-cl-embeddings prophet_extra_inputs/cmp_residual_gated_graph_all64_native512_zscore__plus_tcga_gat_mean300.csv \
  --standard-cl-dim 512 \
  --minimum-gene-absolute-skewness 1.5 \
  --genes-per-skewness-bin 500 \
  --gene-subset-seed 2024 \
  --inverse-response-weighted-mse \
  --inverse-response-epsilon 0.05 \
  --inverse-response-max-weight 20.0 \
  --output-dir results/standard_prophet/CL_skew15_inverse_mse

python3 per_intervention_metrics.py \
  --predictions results/standard_prophet/CL_skew15_inverse_mse/test_predictions.csv \
  --output-dir results/standard_prophet/CL_skew15_inverse_mse
```

Run `python3 train_standard_prophet.py --help` or
`python3 per_intervention_metrics.py --help` for the full flag lists —
see `README.md` for what each group of flags does.

## 6. Example run: the env-var wrapper

`run_standard_prophet.sh` does both steps above in one call, built from
env vars in the same style as the original `submit_tcga_graph_pipeline.sh
transfer` invocation — the same command you were already running maps
onto it almost unchanged, minus `STANDARD_PROPHET_ONLY` (there's only one
mode now) and `--drug-targets` (unused by standard Prophet):

```bash
source prophet_env/bin/activate

PROPHET_GDSC_DATA=prophet_extra_inputs/GDSC2_dataset_prophet.csv \
PROPHET_AUXILIARY_DATA=prophet_extra_inputs/SCORE2_dataset.csv \
IV_EMBEDDINGS=prophet_extra_inputs/iv_embeddings.csv \
STANDARD_CL_EMBEDDINGS=**prophet_assets/embeddings/joint_cl_org_prophet_embeddings/rank_corrected_joint_novel_edges_progeny14_expression512__plus_mutation_gat_mean300.csv** \
STANDARD_CL_DIM=512 \
PROPHET_MIN_GENE_ABS_SKEWNESS=1.5 \
PROPHET_GENES_PER_SKEWNESS_BIN=500 \
PROPHET_GENE_SUBSET_SEED=2024 \
INVERSE_RESPONSE_WEIGHTED_MSE=1 \
INVERSE_RESPONSE_EPSILON=0.05 \
INVERSE_RESPONSE_MAX_WEIGHT=20.0 \
OUTPUT_DIR=results/standard_prophet/CL_skew15_inverse_mse \
bash run_standard_prophet.sh
```

### Env vars the wrapper reads

**Required**: `IV_EMBEDDINGS`, `STANDARD_CL_EMBEDDINGS`, `OUTPUT_DIR`
(the script exits immediately with a clear error if any are unset).

**Data paths** (`PROPHET_GDSC_DATA` defaults to
`prophet_assets/datasets/GDSC2_dataset_prophet.csv` if unset; the other
three are optional and simply omitted from `--prophet-data` if unset —
`--score-only` via `SCORE_ONLY=1` skips the GDSC path entirely, matching
`train_standard_prophet.py`'s own `--score-only`):
`PROPHET_GDSC_DATA` `PROPHET_AUXILIARY_DATA` `PROPHET_ORGANOID_DRUG_DATA`
`PROPHET_ORGANOID_CRISPR_DATA` `SCORE_ONLY`

**Embedding/skewness config**: `STANDARD_CL_DIM`
`EXPECTED_STANDARD_CL_DIM` `GENE_SKEWNESS_PATH` `DRUG_SKEWNESS_PATH`
`PROPHET_MIN_GENE_ABS_SKEWNESS` `PROPHET_GENES_PER_SKEWNESS_BIN`
`PROPHET_GENE_SUBSET_SEED` `PROPHET_MAX_DRUG_SKEWNESS`
`PROPHET_MIN_DRUG_ABS_SKEWNESS`

**Weighted MSE**: `INVERSE_RESPONSE_WEIGHTED_MSE` (`1`/`0`)
`INVERSE_RESPONSE_EPSILON` `INVERSE_RESPONSE_MAX_WEIGHT`
`SKEWNESS_WEIGHTED_MSE` (`1`/`0`) `SKEWNESS_LOSS_STRENGTH`
`SKEWNESS_LOSS_MAX_WEIGHT` `COMBINED_RESPONSE_MAX_WEIGHT`

**Row selection**: `PROTECTED_SUBSAMPLE_PHENOTYPES`
`PROTECTED_SUBSAMPLE_TARGETS` (space-separated lists, unquoted) 
`PHENOTYPE_SAMPLING` `MAX_TRAIN_ROWS` `MAX_VAL_ROWS` `MAX_TEST_ROWS`
`GDSC_MAX_ROWS` `AUXILIARY_MAX_ROWS` `GDSC_FRACTION` `AUXILIARY_FRACTION`

**Model/training**: `BATCH_SIZE` `MAX_STEPS` `LEARNING_RATE`
`WEIGHT_DECAY` `MODEL_DIM` `NUM_HEADS` `NUM_LAYERS` `PATIENCE` `SEED`
`NUM_WORKERS`

**Per-intervention step**: `IV_COL` `CELL_LINE_COL` `PHENOTYPE_COL`
`MIN_CELL_LINES`

**Provenance only** (recorded into `run_config.json`, not passed to
Python — see §7): `MEMORY_MB` `GPU_REQ`

Anything not listed above uses `train_standard_prophet.py`'s own default.

## 7. Farm submission

`run_standard_prophet.sh` runs everything in the foreground in whatever
Python environment is already active — it does not call `bsub` itself.
To submit it to LSF, wrap the call:

```bash
bsub -M "$MEMORY_MB" -R "rusage[mem=${MEMORY_MB}] select[mem>${MEMORY_MB}]" \
     -gpu "$GPU_REQ" -q gpu-normal \
     bash run_standard_prophet.sh
```

with the same env vars exported beforehand. `MEMORY_MB`/`GPU_REQ` are read
by the wrapper only to record them into `run_config.json` for provenance
— set them however your cluster submission needs them, the wrapper
doesn't validate or use their values itself.
