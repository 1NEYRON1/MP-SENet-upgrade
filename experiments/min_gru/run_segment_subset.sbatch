#!/bin/bash
#SBATCH --job-name=mpnet_seg
#SBATCH --partition=gpu
#SBATCH --error=logs/mpnet_seg-%j.err
#SBATCH --output=logs/mpnet_seg-%j.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

# Эксперименты с segment_size и объёмом данных. Одна модель за запуск.
# Примеры (16kHz: 0.5s=8000, 0.8s=12800, 1.0s=16000):
#   MODEL=mpnet SEGMENT_SIZE=8000 OUTPUT_SUFFIX=_0.5s sbatch run_segment_subset.sbatch
#   MODEL=mpnet SEGMENT_SIZE=12800 OUTPUT_SUFFIX=_0.8s sbatch run_segment_subset.sbatch
#   MODEL=mpnet SEGMENT_SIZE=16000 OUTPUT_SUFFIX=_1.0s sbatch run_segment_subset.sbatch

mkdir -p logs checkpoints

# Activate venv (created by setup_env.sh in experiments/)
source ../.venv/bin/activate

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

MODEL=${MODEL:-mpnet}
: ${OUTPUT_SUFFIX:?OUTPUT_SUFFIX required, e.g. OUTPUT_SUFFIX=_0.5s}
EXTRA=""
[ -n "${SEGMENT_SIZE:-}" ] && EXTRA="$EXTRA --segment_size $SEGMENT_SIZE"
[ -n "${SUBSET_RATIO:-}" ] && EXTRA="$EXTRA --subset_ratio $SUBSET_RATIO"

echo "Model: $MODEL"
echo "Output: checkpoints/${MODEL}${OUTPUT_SUFFIX}"
echo "Extra: $EXTRA"
echo "=========================================="

python -u run_experiment.py \
    --model $MODEL \
    --epochs 100 \
    --save_every 1 \
    --data_dir ../VoiceBank+DEMAND \
    --output_dir "checkpoints/${MODEL}${OUTPUT_SUFFIX}" \
    $EXTRA
