#!/bin/bash

#SBATCH --job-name=ivsn_fovea_compare
#SBATCH --output=logs/ivsn_fovea_compare_%j.out
#SBATCH --error=logs/ivsn_fovea_compare_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

mkdir -p logs
source venv/bin/activate

DATA_ROOT="data/coco_crops_transparent_8cat"
BASE_OUT="outputs/vgg_8obj_rotation_compare"

COMMON_ARGS="\
  --data-root ${DATA_ROOT} \
  --model-kind vgg \
  --n-objects 8 \
  --transform-mode rotation \
  --smoothing-mode alpha \
  --smooth-target \
  --smooth-cue \
  --smooth-distractors"

echo "Running uniform / no peripheral blur baseline..."

python fovea_ivsn_run.py \
  ${COMMON_ARGS} \
  --out-dir "${BASE_OUT}/uniform"

MANIFEST=$(find "${BASE_OUT}/uniform" -name "base_trials_manifest.json" | head -n 1)

echo "Using base manifest:"
echo "${MANIFEST}"

echo "Running foveated / peripheral blur condition..."

python fovea_ivsn_run.py \
  ${COMMON_ARGS} \
  --out-dir "${BASE_OUT}/foveated" \
  --load-base-manifest "${MANIFEST}" \
  --foveated-search \
  --fovea-radius 90 \
  --fovea-transition-width 120 \
  --periphery-blur-radius 6

echo "Done."