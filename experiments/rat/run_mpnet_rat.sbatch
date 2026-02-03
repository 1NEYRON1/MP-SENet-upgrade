#!/bin/bash
#SBATCH --job-name=mpnet_rat             # MPNet + MPNet RAT experiments
#SBATCH --partition=gpu
#SBATCH --error=logs/mpnet-%A-%a.err
#SBATCH --output=logs/mpnet-%A-%a.log
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --array=0-5                      # 0=mpnet, 1-5=mpnet_rat chunk 4,8,16,32,64

mkdir -p logs checkpoints

# Activate venv (created by setup_env.sh in experiments/)
source ../.venv/bin/activate

# Add CUDA to path if available
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

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
    MODEL="mpnet"
    CHUNK=""
else
    MODEL="mpnet_rat"
    CHUNK="--chunk_size ${CHUNK_SIZES[$TASK_ID-1]}"
fi

echo "Model: $MODEL $CHUNK"
echo "=========================================="

python -u run_experiment.py \
    --model $MODEL \
    $CHUNK \
    --epochs 100 \
    --batch_size 4 \
    --num_workers 4 \
    --save_every 10 \
    --data_dir ../VoiceBank+DEMAND \
    --output_dir checkpoints
