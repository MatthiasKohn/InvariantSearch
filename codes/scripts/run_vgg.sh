#!/bin/bash

#SBATCH --job-name=single_run
#SBATCH --output=logs/train_%j.out     # Output log (%j = job ID)
#SBATCH --error=logs/train_%j.err      # Error logs
#SBATCH --time=00:30:00                # Max runtime(30 Mins)
#SBATCH --partition=gpu_a100_short     # short for jobs with max 30 min
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

source venv/bin/activate

mkdir -p logs
SCRIPT="ivsn_invariant_search.py"
RESULT_ROOT="results"
DATA="data/coco_crops_transparent_8cat"

COMMON_ARGS=(
  --model-kind vgg
  --smoothing-mode alpha
  --smooth-target
  --smooth-cue
  --smooth-distractors
)

run_set () {
  local NOBJ="$1"
  local DATA_ROOT="$2"

  python "$SCRIPT" --data-root "$DATA_ROOT" --out-dir "$RESULT_ROOT/vgg_${NOBJ}obj_baseline" \
    --n-objects "$NOBJ" --transform-mode scale --scale-values 1.0 "${COMMON_ARGS[@]}"

  python "$SCRIPT" --data-root "$DATA_ROOT" --out-dir "$RESULT_ROOT/vgg_${NOBJ}obj_rotation" \
    --n-objects "$NOBJ" --transform-mode rotation --rotation-values 0 30 60 90 120 150 180 "${COMMON_ARGS[@]}"

  python "$SCRIPT" --data-root "$DATA_ROOT" --out-dir "$RESULT_ROOT/vgg_${NOBJ}obj_scale" \
    --n-objects "$NOBJ" --transform-mode scale --scale-values 0.5 0.75 1.0 1.25 1.5 "${COMMON_ARGS[@]}"


  python "$SCRIPT" --data-root "$DATA_ROOT" --out-dir "$RESULT_ROOT/vgg_${NOBJ}obj_skewx" \
    --n-objects "$NOBJ" --transform-mode skew_x --skew-values -20 -10 0 10 20 "${COMMON_ARGS[@]}"

  python "$SCRIPT" --data-root "$DATA_ROOT" --out-dir "$RESULT_ROOT/vgg_${NOBJ}obj_skewy" \
    --n-objects "$NOBJ" --transform-mode skew_y --skew-values -20 -10 0 10 20 "${COMMON_ARGS[@]}"

  python "$SCRIPT" --data-root "$DATA_ROOT" --out-dir "$RESULT_ROOT/vgg_${NOBJ}obj_noise" \
    --n-objects "$NOBJ" --transform-mode noise --noise-values 0.0 0.03 0.06 0.09 0.12 "${COMMON_ARGS[@]}"

  python "$SCRIPT" --data-root "$DATA_ROOT" --out-dir "$RESULT_ROOT/vgg_${NOBJ}obj_blur" \
    --n-objects "$NOBJ" --transform-mode blur --blur-values 0.0 0.5 1.0 2.0 3.0 "${COMMON_ARGS[@]}"
}

run_set 6 "$DATA"
run_set 8 "$DATA"
