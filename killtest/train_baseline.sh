#!/bin/bash
# T1.3 baseline trainings for the bimodal-depth kill test.
# usage: train_baseline.sh <scene_dir> <arm> <gpu_id> [extra ns-train args...]
#   arms:
#     edgeaware  - repo-default unimodal depth loss (EdgeAwareLogL1, RGB-grad downweighted)
#     l1         - pure unimodal L1 depth loss (no edge leniency)
#     nodepth    - depth supervision disabled (photometric + normal + smooth only)
# All other losses (normal, depth-smooth TV, scale) identical across arms.
# Depth source: MuSHRoom iphone sensor depth (dataparser default depth_mode=sensor).
set -e
scene_dir=$1
arm=$2
gpu=$3
shift 3
scene=$(basename "$scene_dir")

outdir=/scratch/adam/experiments/bimodal-depth/outputs

case $arm in
  edgeaware)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type EdgeAwareLogL1 --pipeline.model.depth-lambda 0.2"
    ;;
  l1)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type L1 --pipeline.model.depth-lambda 0.2"
    ;;
  nodepth)
    depth_flags="--pipeline.model.use-depth-loss False"
    ;;
  logl1)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type LogL1 --pipeline.model.depth-lambda 0.2"
    ;;
  masked)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type MaskedLogL1 --pipeline.model.depth-lambda 0.2"
    ;;
  bimodal)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type BimodalLogL1 --pipeline.model.depth-lambda 0.2"
    ;;
  soft)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type SoftBimodalLogL1 --pipeline.model.depth-lambda 0.2"
    ;;
  masked4px)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type MaskedLogL1 --pipeline.model.depth-lambda 0.2"
    export DN_HYP_DIR=bimodal_hyp_4px
    ;;
  bimodal4px)
    depth_flags="--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type BimodalLogL1 --pipeline.model.depth-lambda 0.2"
    export DN_HYP_DIR=bimodal_hyp_4px
    ;;
  *)
    echo "unknown arm: $arm"; exit 1
    ;;
esac

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dnsplat
# gsplat 1.0.0 JIT-compiles its CUDA extension at import; needs nvcc + arch
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0
cd ~/repos/dn-splatter

CUDA_VISIBLE_DEVICES=$gpu ns-train dn-splatter \
  --output-dir "$outdir" \
  --experiment-name "${scene}_${arm}" \
  --timestamp run1 \
  --vis tensorboard \
  --pipeline.model.use-depth-smooth-loss True \
  --pipeline.model.use-normal-loss True \
  --pipeline.model.normal-supervision mono \
  $depth_flags \
  "$@" \
  mushroom --data "$scene_dir" --mode iphone

echo "TRAIN_DONE ${scene}_${arm}"
