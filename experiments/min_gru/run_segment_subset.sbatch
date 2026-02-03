#!/bin/bash
#SBATCH --job-name=mpnet_seg
#SBATCH --partition=gpu
#SBATCH --error=logs/mpnet_seg-%j.err
#SBATCH --output=logs/mpnet_seg-%j.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=jupyterhub-slurm-node2

# Эксперименты с segment_size и объёмом данных. Одна модель за запуск.
# Примеры:
#   MODEL=min_mpnet SEGMENT_SIZE=16000 OUTPUT_SUFFIX=_seg16k_min sbatch run_segment_subset.sbatch
#   MODEL=mpnet SEGMENT_SIZE=64000 OUTPUT_SUFFIX=_seg64k sbatch run_segment_subset.sbatch
#   MODEL=min_mpnet SUBSET_RATIO=0.5 OUTPUT_SUFFIX=_subset_0.5 sbatch run_segment_subset.sbatch

mkdir -p logs checkpoints

source .venv/bin/activate

MODEL=${MODEL:-mpnet}
: ${OUTPUT_SUFFIX:?OUTPUT_SUFFIX required, e.g. OUTPUT_SUFFIX=_seg16k}
EXTRA=""
[ -n "${SEGMENT_SIZE:-}" ] && EXTRA="$EXTRA --segment_size $SEGMENT_SIZE"
[ -n "${SUBSET_RATIO:-}" ] && EXTRA="$EXTRA --subset_ratio $SUBSET_RATIO"

echo "Model: $MODEL"
echo "Output: checkpoints/${MODEL}${OUTPUT_SUFFIX}"
echo "Extra: $EXTRA"
echo "=========================================="

python -u run_experiment.py \
    --model $MODEL \
    --epochs 400 \
    --save_every 10 \
    --data_dir ../../VoiceBank+DEMAND \
    --output_dir "checkpoints/${MODEL}${OUTPUT_SUFFIX}" \
    $EXTRA
