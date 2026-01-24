#!/bin/bash
#SBATCH --job-name=mingru_exp               # Название задачи
#SBATCH --partition=gpu                     # GPU очередь
#SBATCH --error=logs/exp-%A-%a.err          # Файл ошибок (%A=job_id, %a=array_id)
#SBATCH --output=logs/exp-%A-%a.log         # Файл вывода
#SBATCH --time=168:00:00                    # 7 дней
#SBATCH --nodes=1                           # 1 узел
#SBATCH --ntasks=1                          # 1 MPI процесс
#SBATCH --cpus-per-task=8                   # 8 CPU (для DataLoader)
#SBATCH --gpus-per-task=1                   # 1 GPU на эксперимент
#SBATCH --array=0-1                         # GTCRN эксперименты (0-1)

mkdir -p logs checkpoints

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

# GTCRN Experiments (1 GPU):
# 0: gtcrn (baseline)
# 1: min_gtcrn (with MinGRU)

MODELS=("gtcrn" "min_gtcrn")
TASK_ID=$SLURM_ARRAY_TASK_ID

MODEL=${MODELS[$TASK_ID]}

echo "Model: $MODEL"
echo "=========================================="

python -u run_experiment.py \
    --model $MODEL \
    --epochs 100 \
    --save_every 10 \
    --data_dir ../../VoiceBank+DEMAND \
    --output_dir checkpoints
