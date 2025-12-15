#!/bin/bash
#SBATCH --job-name=mingru_exp            # Название задачи
#SBATCH --partition=normal               # Основная очередь (поддерживает GPU)
#SBATCH --error=logs/exp-%A-%a.err       # Файл ошибок (%A=job_id, %a=array_id)
#SBATCH --output=logs/exp-%A-%a.log      # Файл вывода
#SBATCH --time=24:00:00                  # Максимальное время (до 30 дней в normal)
#SBATCH --gpus=1                         # 1 GPU на эксперимент
#SBATCH --cpus-per-task=4                # 4 CPU
#SBATCH --nodes=1                        # 1 узел
#SBATCH --array=0-3                      # 4 эксперимента (0-3)

mkdir -p logs checkpoints

module purge
module load Python
module load CUDA/12.4

eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null
conda activate gtcrn_env

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

# Experiments:
# 0: gtcrn (baseline)
# 1: min_gtcrn (with MinGRU)
# 2: mpnet (baseline)
# 3: min_mpnet (with MinGRU)

MODELS=("gtcrn" "min_gtcrn" "mpnet" "min_mpnet")
EPOCHS=(100 100 400 400)  # gtcrn: 100, mpnet: 400
TASK_ID=$SLURM_ARRAY_TASK_ID

MODEL=${MODELS[$TASK_ID]}
NUM_EPOCHS=${EPOCHS[$TASK_ID]}

echo "Model: $MODEL"
echo "Epochs: $NUM_EPOCHS"
echo "=========================================="

python -u run_experiment.py \
    --model $MODEL \
    --epochs $NUM_EPOCHS \
    --save_every 10 \
    --data_dir ../../VoiceBank+DEMAND \
    --output_dir checkpoints
