#!/bin/bash
# Full T1.3 baseline queue: 3 arms x 3 scenes, split across 2 GPUs.
# usage: run_queue.sh <gpu_id> <"scene:arm scene:arm ...">
# Renders the eval split after each training.
set -e
gpu=$1
shift
jobs="$@"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dnsplat
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0
# our own checkpoints are trusted; torch>=2.6 weights_only default breaks ns loads
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

E=/scratch/adam/experiments/bimodal-depth
DATA=/scratch/adam/datasets/mushroom/room_datasets

# wait for asset pregeneration (normals/pointclouds for coffee_room + vr_room)
if [ -z "$SKIP_PREGEN_WAIT" ]; then
  until grep -q PREGEN_DONE $E/pregen_rest.log 2>/dev/null; do sleep 30; done
fi

cd $E
for job in $jobs; do
  scene=${job%%:*}
  arm=${job##*:}
  echo "=== $(date +%H:%M:%S) starting ${scene}_${arm} on gpu$gpu ==="
  ~/repos/dn-splatter/killtest/train_baseline.sh $DATA/$scene $arm $gpu \
    > $E/train_${scene}_${arm}.log 2>&1
  echo "=== $(date +%H:%M:%S) rendering ${scene}_${arm} ==="
  CUDA_VISIBLE_DEVICES=$gpu python -u ~/repos/dn-splatter/dn_splatter/scripts/render_model.py \
    --load-config $E/outputs/${scene}_${arm}/dn-splatter/run1/config.yml \
    --output-dir renders/${scene}_${arm} --split eval \
    > $E/render_${scene}_${arm}.log 2>&1
  echo "=== DONE ${scene}_${arm} ==="
done
echo QUEUE_DONE_GPU$gpu
