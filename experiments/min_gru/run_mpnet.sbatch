#!/bin/bash
#SBATCH --job-name=mpnet_exp                # Название задачи
#SBATCH --partition=gpu                     # GPU очередь
#SBATCH --error=logs/mpnet-%A-%a.err        # Файл ошибок
#SBATCH --output=logs/mpnet-%A-%a.log       # Файл вывода
#SBATCH --time=168:00:00                    # 7 дней
#SBATCH --nodes=1                           # 1 узел
#SBATCH --ntasks=1                          # 1 MPI процесс
#SBATCH --cpus-per-task=16                  # 16 CPU (для DataLoader)
#SBATCH --gpus-per-task=1                   # 1 GPU для MPNet
#SBATCH --array=0-1                         # MPNet эксперименты (0-1)
#SBATCH --nodelist=jupyterhub-slurm-node2   # H200 node
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

# MPNet Experiments (1 GPU):
# 0: mpnet (baseline)
# 1: min_mpnet (with MinGRU)

MODELS=("mpnet" "min_mpnet")
TASK_ID=$SLURM_ARRAY_TASK_ID

MODEL=${MODELS[$TASK_ID]}

echo "Model: $MODEL"
echo "=========================================="

python -u run_experiment.py \
    --model $MODEL \
    --epochs 400 \
    --save_every 10 \
    --data_dir ../../VoiceBank+DEMAND \
    --output_dir checkpoints

