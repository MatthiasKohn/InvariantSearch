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

python fovea_ivsn_run.py \
  --data-root "C:\Users\kohnm\Uni\Visual-Search\data\coco_crops_transparent_8cat" \
  --out-dir outputs/vgg_8obj_rotation \
  --model-kind vgg \
  --n-objects 8 \
  --transform-mode rotation \
  --smoothing-mode alpha \
  --smooth-target \
  --smooth-cue \
  --smooth-distractors \
  --foveated-search \
  --fovea-radius 90 \
  --fovea-transition-width 120 \
  --periphery-blur-radius 6
