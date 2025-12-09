#!/bin/bash
#SBATCH --job-name=gtcrn_exp             # Название задачи
#SBATCH --partition=normal               # Основная очередь (поддерживает GPU)
#SBATCH --error=logs/exp-%A-%a.err       # Файл ошибок (%A=job_id, %a=array_id)
#SBATCH --output=logs/exp-%A-%a.log      # Файл вывода
#SBATCH --time=24:00:00                  # Максимальное время (до 30 дней в normal)
#SBATCH --gpus=1                         # 1 GPU на эксперимент
#SBATCH --cpus-per-task=4                # 4 CPU
#SBATCH --nodes=1                        # 1 узел
#SBATCH --array=0-10                     # 11 экспериментов (0-10)

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

CHUNK_SIZES=(4 8 16 32 64)
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ $TASK_ID -eq 0 ]; then
    MODEL="gtcrn"
    CHUNK=""
elif [ $TASK_ID -le 5 ]; then
    MODEL="gtcrn_rat"
    CHUNK="--chunk_size ${CHUNK_SIZES[$TASK_ID-1]}"
else
    MODEL="gtcrn_grat"
    CHUNK="--chunk_size ${CHUNK_SIZES[$TASK_ID-6]}"
fi

echo "Model: $MODEL $CHUNK"
echo "=========================================="

python -u run_experiment.py \
    --model $MODEL \
    $CHUNK \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-3 \
    --num_workers 4 \
    --save_every 10 \
    --data_dir data \
    --output_dir checkpoints
