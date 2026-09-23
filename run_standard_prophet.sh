#!/usr/bin/env bash
# Env-var-driven wrapper around train_standard_prophet.py + per_intervention_metrics.py.
#
# This does three things in sequence:
#   1. Assembles --prophet-data from whichever PROPHET_*_DATA paths are set
#      (GDSC is always first; auxiliary/organoid datasets are optional).
#   2. Runs train_standard_prophet.py with those paths and every other
#      PROPHET_*/INVERSE_RESPONSE_*/SKEWNESS_* env var mapped onto its flags.
#   3. Runs per_intervention_metrics.py against the resulting
#      test_predictions.csv and writes per_intervention_metrics.csv into the
#      same --output-dir.
#
# It runs everything in the foreground, in whatever Python environment is
# already active (activate prophet_env yourself first). If you want to
# submit this to LSF instead of running it directly, wrap the call itself,
# e.g.:
#   bsub -M "$MEMORY_MB" -R "rusage[mem=${MEMORY_MB}] select[mem>${MEMORY_MB}]" \
#        -gpu "$GPU_REQ" -q gpu-normal \
#        bash run_standard_prophet.sh
# (MEMORY_MB/GPU_REQ are read below only to record them into run_config.json
# for provenance -- this script does not call bsub itself.)

set -euo pipefail

: "${PROPHET_GDSC_DATA:=prophet_assets/datasets/GDSC2_dataset_prophet.csv}"
: "${IV_EMBEDDINGS:?Set IV_EMBEDDINGS to the intervention embedding CSV}"
: "${STANDARD_CL_EMBEDDINGS:?Set STANDARD_CL_EMBEDDINGS to the cell-line embedding CSV}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

# --- 1. Assemble --prophet-data --------------------------------------------
prophet_data_paths=("$PROPHET_GDSC_DATA")
[ -n "${PROPHET_AUXILIARY_DATA:-}" ] && prophet_data_paths+=("$PROPHET_AUXILIARY_DATA")
[ -n "${PROPHET_ORGANOID_DRUG_DATA:-}" ] && prophet_data_paths+=("$PROPHET_ORGANOID_DRUG_DATA")
[ -n "${PROPHET_ORGANOID_CRISPR_DATA:-}" ] && prophet_data_paths+=("$PROPHET_ORGANOID_CRISPR_DATA")

for path in "${prophet_data_paths[@]}"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: prophet-data path does not exist: $path" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

# --- 2. Snapshot the resolved config for provenance ------------------------
cat > "$OUTPUT_DIR/run_config.json" <<JSON
{
  "prophet_data_paths": $(printf '%s\n' "${prophet_data_paths[@]}" | python3 -c "import sys, json; print(json.dumps([line.strip() for line in sys.stdin]))"),
  "iv_embeddings": "$IV_EMBEDDINGS",
  "standard_cl_embeddings": "$STANDARD_CL_EMBEDDINGS",
  "standard_cl_dim": "${STANDARD_CL_DIM:-null}",
  "minimum_gene_absolute_skewness": "${PROPHET_MIN_GENE_ABS_SKEWNESS:-0.0}",
  "genes_per_skewness_bin": "${PROPHET_GENES_PER_SKEWNESS_BIN:-0}",
  "gene_subset_seed": "${PROPHET_GENE_SUBSET_SEED:-2024}",
  "maximum_drug_skewness": "${PROPHET_MAX_DRUG_SKEWNESS:-null}",
  "inverse_response_weighted_mse": "${INVERSE_RESPONSE_WEIGHTED_MSE:-0}",
  "inverse_response_epsilon": "${INVERSE_RESPONSE_EPSILON:-0.05}",
  "inverse_response_max_weight": "${INVERSE_RESPONSE_MAX_WEIGHT:-10.0}",
  "skewness_weighted_mse": "${SKEWNESS_WEIGHTED_MSE:-0}",
  "output_dir": "$OUTPUT_DIR",
  "memory_mb": "${MEMORY_MB:-null}",
  "gpu_req": "${GPU_REQ:-null}"
}
JSON

# --- 3. Build the train_standard_prophet.py argument list ------------------
train_args=(
    --prophet-data "${prophet_data_paths[@]}"
    --iv-embeddings "$IV_EMBEDDINGS"
    --standard-cl-embeddings "$STANDARD_CL_EMBEDDINGS"
    --output-dir "$OUTPUT_DIR"
)

[ -n "${STANDARD_CL_DIM:-}" ] && train_args+=(--standard-cl-dim "$STANDARD_CL_DIM")
[ -n "${EXPECTED_STANDARD_CL_DIM:-}" ] && train_args+=(--expected-standard-cl-dim "$EXPECTED_STANDARD_CL_DIM")
[ -n "${GENE_SKEWNESS_PATH:-}" ] && train_args+=(--gene-skewness-path "$GENE_SKEWNESS_PATH")
[ -n "${DRUG_SKEWNESS_PATH:-}" ] && train_args+=(--drug-skewness-path "$DRUG_SKEWNESS_PATH")
[ -n "${PROPHET_MIN_GENE_ABS_SKEWNESS:-}" ] && train_args+=(--minimum-gene-absolute-skewness "$PROPHET_MIN_GENE_ABS_SKEWNESS")
[ -n "${PROPHET_GENES_PER_SKEWNESS_BIN:-}" ] && train_args+=(--genes-per-skewness-bin "$PROPHET_GENES_PER_SKEWNESS_BIN")
[ -n "${PROPHET_GENE_SUBSET_SEED:-}" ] && train_args+=(--gene-subset-seed "$PROPHET_GENE_SUBSET_SEED")
[ -n "${PROPHET_MAX_DRUG_SKEWNESS:-}" ] && train_args+=(--maximum-drug-skewness "$PROPHET_MAX_DRUG_SKEWNESS")
[ -n "${PROPHET_MIN_DRUG_ABS_SKEWNESS:-}" ] && train_args+=(--minimum-drug-absolute-skewness "$PROPHET_MIN_DRUG_ABS_SKEWNESS")

if [ "${INVERSE_RESPONSE_WEIGHTED_MSE:-0}" = "1" ]; then
    train_args+=(--inverse-response-weighted-mse)
    [ -n "${INVERSE_RESPONSE_EPSILON:-}" ] && train_args+=(--inverse-response-epsilon "$INVERSE_RESPONSE_EPSILON")
    [ -n "${INVERSE_RESPONSE_MAX_WEIGHT:-}" ] && train_args+=(--inverse-response-max-weight "$INVERSE_RESPONSE_MAX_WEIGHT")
fi
if [ "${SKEWNESS_WEIGHTED_MSE:-0}" = "1" ]; then
    train_args+=(--skewness-weighted-mse)
    [ -n "${SKEWNESS_LOSS_STRENGTH:-}" ] && train_args+=(--skewness-loss-strength "$SKEWNESS_LOSS_STRENGTH")
    [ -n "${SKEWNESS_LOSS_MAX_WEIGHT:-}" ] && train_args+=(--skewness-loss-max-weight "$SKEWNESS_LOSS_MAX_WEIGHT")
fi
[ -n "${COMBINED_RESPONSE_MAX_WEIGHT:-}" ] && train_args+=(--combined-response-max-weight "$COMBINED_RESPONSE_MAX_WEIGHT")

[ -n "${PROTECTED_SUBSAMPLE_PHENOTYPES:-}" ] && train_args+=(--protected-subsample-phenotypes $PROTECTED_SUBSAMPLE_PHENOTYPES)
[ -n "${PROTECTED_SUBSAMPLE_TARGETS:-}" ] && train_args+=(--protected-subsample-targets $PROTECTED_SUBSAMPLE_TARGETS)
[ -n "${PHENOTYPE_SAMPLING:-}" ] && train_args+=(--phenotype-sampling "$PHENOTYPE_SAMPLING")
[ -n "${MAX_TRAIN_ROWS:-}" ] && train_args+=(--max-train-rows "$MAX_TRAIN_ROWS")
[ -n "${MAX_VAL_ROWS:-}" ] && train_args+=(--max-val-rows "$MAX_VAL_ROWS")
[ -n "${MAX_TEST_ROWS:-}" ] && train_args+=(--max-test-rows "$MAX_TEST_ROWS")
[ -n "${GDSC_MAX_ROWS:-}" ] && train_args+=(--gdsc-max-rows "$GDSC_MAX_ROWS")
[ -n "${AUXILIARY_MAX_ROWS:-}" ] && train_args+=(--auxiliary-max-rows "$AUXILIARY_MAX_ROWS")
[ -n "${GDSC_FRACTION:-}" ] && train_args+=(--gdsc-fraction "$GDSC_FRACTION")
[ -n "${AUXILIARY_FRACTION:-}" ] && train_args+=(--auxiliary-fraction "$AUXILIARY_FRACTION")
[ "${SCORE_ONLY:-0}" = "1" ] && train_args+=(--score-only)

[ -n "${BATCH_SIZE:-}" ] && train_args+=(--batch-size "$BATCH_SIZE")
[ -n "${MAX_STEPS:-}" ] && train_args+=(--max-steps "$MAX_STEPS")
[ -n "${LEARNING_RATE:-}" ] && train_args+=(--learning-rate "$LEARNING_RATE")
[ -n "${WEIGHT_DECAY:-}" ] && train_args+=(--weight-decay "$WEIGHT_DECAY")
[ -n "${MODEL_DIM:-}" ] && train_args+=(--model-dim "$MODEL_DIM")
[ -n "${NUM_HEADS:-}" ] && train_args+=(--num-heads "$NUM_HEADS")
[ -n "${NUM_LAYERS:-}" ] && train_args+=(--num-layers "$NUM_LAYERS")
[ -n "${PATIENCE:-}" ] && train_args+=(--patience "$PATIENCE")
[ -n "${SEED:-}" ] && train_args+=(--seed "$SEED")
[ -n "${NUM_WORKERS:-}" ] && train_args+=(--num-workers "$NUM_WORKERS")

echo "Running: python3 train_standard_prophet.py ${train_args[*]}"
python3 train_standard_prophet.py "${train_args[@]}"

# --- 4. Per-intervention metrics --------------------------------------------
predictions_csv="$OUTPUT_DIR/test_predictions.csv"
if [ ! -f "$predictions_csv" ]; then
    echo "ERROR: $predictions_csv was not produced by training -- skipping per-intervention metrics." >&2
    exit 1
fi

iv_col="${IV_COL:-iv1}"
cell_line_col="${CELL_LINE_COL:-cell_line}"
phenotype_col="${PHENOTYPE_COL:-phenotype}"
min_cell_lines="${MIN_CELL_LINES:-3}"

echo "Running: python3 per_intervention_metrics.py --predictions $predictions_csv --output-dir $OUTPUT_DIR"
python3 per_intervention_metrics.py \
    --predictions "$predictions_csv" \
    --output-dir "$OUTPUT_DIR" \
    --iv-col "$iv_col" \
    --cell-line-col "$cell_line_col" \
    --phenotype-col "$phenotype_col" \
    --min-cell-lines "$min_cell_lines"

echo "Done. See $OUTPUT_DIR/summary.json and $OUTPUT_DIR/per_intervention_metrics.csv"
