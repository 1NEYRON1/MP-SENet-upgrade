#!/bin/bash
echo "=== Setting up gtcrn_env environment ==="

module purge
module load Python

echo "Creating conda environment..."
conda create -n gtcrn_env python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate gtcrn_env
conda install -c conda-forge soxr-python -y

echo "Installing PyTorch with CUDA 12.4 support..."
module load CUDA/12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing dependencies..."
CFLAGS="-std=c99" pip3 install soundfile pesq tqdm matplotlib numpy einops librosa scipy joblib

echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import soundfile; print('soundfile: OK')"
python -c "from pesq import pesq; print('pesq: OK')"
python -c "import tqdm; print('tqdm: OK')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import einops; print('einops: OK')"
python -c "import librosa; print('librosa: OK')"
python -c "import scipy; print('scipy: OK')"
python -c "import joblib; print('joblib: OK')"

echo "=== Done! ==="
echo "Next steps:"
echo "  1. source activate gtcrn_env"
echo "  2. sbatch run_all.sbatch      # Запустить ВСЕ эксперименты"
echo "  3. sbatch run_single.sbatch gtcrn # Или один эксперимент"
echo "  4. mj                         # Проверить статус задач"
