#!/bin/bash
#SBATCH --job-name=senet_train
#SBATCH --output=../comp_logs/senet_%j.out
#SBATCH --error=../comp_logs/senet_%j.out
#SBATCH --gpus=1
#SBATCH --time=20:30:00
#SBATCH --cpus-per-task=6
#SBATCH --chdir=/home2/dmmartsenyuk/git_proj
#SBATCH --nodelist=jupyterhub-slurm-node2   # H200 node
# usage: sbatch run.sbatch --config <path> --save_dir <path>

CONFIG=""
CP="cp_model"
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)   CONFIG="$2"; shift 2 ;;
    --save_dir) CP="$2";    shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
[[ -n "$CONFIG" ]] || { echo "usage: sbatch run.sbatch --config <path> --save_dir <path>"; exit 1; }

echo "Date = $(date)"
echo "Hostname = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Config = $CONFIG"
echo "Checkpoint = $CP"
echo ""
echo "Number of Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

if [[ -d ".venv" ]]; then
  source .venv/bin/activate
else
  echo "Error: .venv not found. Run ./setup.sh on the cluster first."
  exit 1
fi

python -u train_unified.py --checkpoint_path "$CP" --config "$CONFIG"
