#!/bin/bash
#SBATCH --job-name=mpnet_exp                # Название задачи
#SBATCH --partition=normal                  # Основная очередь (поддерживает GPU)
#SBATCH --error=logs/mpnet-%A-%a.err        # Файл ошибок
#SBATCH --output=logs/mpnet-%A-%a.log       # Файл вывода
#SBATCH --time=168:00:00                    # 7 дней
#SBATCH --nodes=1                           # 1 узел
#SBATCH --ntasks=1                          # 1 MPI процесс
#SBATCH --cpus-per-task=16                  # 16 CPU (для DataLoader, больше для 2 GPU)
#SBATCH --gpus=4                           # 2 GPU для MPNet
#SBATCH --constraint="type_a|type_b|type_c" # V100 32GB
#SBATCH --array=0-1                         # MPNet эксперименты (0-1)

mkdir -p logs checkpoints

module purge
module load Python
module load CUDA/12.4

conda activate gtcrn_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

# MPNet Experiments (4 GPU):
# 0: mpnet (baseline)
# 1: min_mpnet (with MinGRU)

MODELS=("mpnet" "min_mpnet")
TASK_ID=$SLURM_ARRAY_TASK_ID

MODEL=${MODELS[$TASK_ID]}

echo "Model: $MODEL"
echo "=========================================="

torchrun --nproc_per_node=4 train_mpnet_ddp.py \
    --model $MODEL \
    --epochs 400 \
    --save_every 10 \
    --data_dir ../../VoiceBank+DEMAND \
    --output_dir checkpoints

