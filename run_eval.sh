#!/bin/bash

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- environment ---
if command -v ml >/dev/null 2>&1; then
  ml conda || true
fi
conda activate "${CONDA_ENV:-traffic-vlm}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

# --- config ---
TAGS=(baseline card_owned)

DATA_DTLD="data/dtld"
DATA_LABELS="data/labels"
DATA_LASER="data/laser"
DATA_CROPS="data/crops_jpg"
DATA_SYNTH="data/synth_jpg"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_ATTACK="${MAX_PER_ATTACK_COMBO:-1000}"
MAX_BENIGN="${MAX_PER_STATE:-1000}"

BASELINE_CKPT="checkpoints/baseline/best_model.pt"

[[ -f "${BASELINE_CKPT}" ]] || { echo "missing ${BASELINE_CKPT}" >&2; exit 1; }
[[ -d "${DATA_SYNTH}/generated" ]] || {
  echo "missing ${DATA_SYNTH}/generated — run process_data.py first" >&2
  exit 1
}

for TAG in "${TAGS[@]}"; do
  CKPT="checkpoints/${TAG}/best_model.pt"
  META="checkpoints/${TAG}/best_model.json"
  [[ -f "${META}" ]] || META="checkpoints/${TAG}/run_meta.json"
  LASER_DIR="output/laser_injection/${TAG}"
  CONS_DIR="output/cardinality_consistency/${TAG}"

  mkdir -p "${LASER_DIR}" "${CONS_DIR}"

  echo "=== ${TAG} ==="
  echo "ckpt=${CKPT}"
  echo "laser_dir=${LASER_DIR}"
  echo "cons_dir=${CONS_DIR}"

  [[ -f "${CKPT}" ]] || { echo "missing ${CKPT}" >&2; exit 1; }

  # --- 1) laser ASR ---
  if [[ -f "${LASER_DIR}/attack_ranking.json" ]]; then
    echo "skip laser (attack_ranking.json exists)"
  else
    echo "=== laser eval ==="
    ATTACK_ARGS=(
      python test_models.py
      --models ontology
      --ontology_model "${CKPT}"
      --baseline_model "${BASELINE_CKPT}"
      --benign_source dtld
      --dtld_root "${DATA_DTLD}"
      --dtld_split train
      --synth_cache_dir "${DATA_SYNTH}"
      --red_laser_dir "${DATA_LASER}/Red"
      --green_laser_dir "${DATA_LASER}/Green"
      --ir_laser_dir "${DATA_LASER}/IR_modulated"
      --skip_module_analysis
      --skip_vlm_samples
      --skip_simulation
      --batch_size "${BATCH_SIZE}"
      --output_dir "${LASER_DIR}"
    )
    [[ "${MAX_ATTACK}" -gt 0 ]] && ATTACK_ARGS+=(--max_per_attack_combo "${MAX_ATTACK}")
    "${ATTACK_ARGS[@]}"
  fi

  # --- 2) cardinality consistency ---
  if [[ -f "${CONS_DIR}/benign/summary.json" && -f "${CONS_DIR}/attack/summary.json" ]]; then
    echo "skip consistency (summaries exist)"
  else
    echo "=== consistency eval ==="
    CONS_ARGS=(
      python eval_cardinality_consistency.py
      --ckpt "${CKPT}"
      --source both
      --image_dir "${DATA_CROPS}"
      --labels_dir "${DATA_LABELS}"
      --laser_dir "${LASER_DIR}"
      --benign_from_laser
      --out_dir "${CONS_DIR}"
      --batch_size "${BATCH_SIZE}"
      --write_details
    )
    [[ "${MAX_BENIGN}" -gt 0 ]] && CONS_ARGS+=(--max_per_state "${MAX_BENIGN}")
    [[ -f "${META}" ]] && CONS_ARGS+=(--meta "${META}")
    "${CONS_ARGS[@]}"
  fi

  echo "=== done ${TAG} ==="
done
