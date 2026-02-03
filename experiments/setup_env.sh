#!/bin/bash
echo "=== Setting up .venv environment ==="

# Create venv
python3 -m venv .venv
source .venv/bin/activate

echo "Installing PyTorch with CUDA 12.4 support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing dependencies..."
CFLAGS="-std=c99" pip install soundfile pesq tqdm matplotlib numpy einops librosa scipy joblib

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
echo "  1. source .venv/bin/activate"
echo "  2. cd rat && sbatch run_mpnet_rat.sbatch"
echo "  3. squeue -u \$USER   # Проверить статус задач"
