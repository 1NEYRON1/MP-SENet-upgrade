#!/bin/bash
set -e

echo "=== Setting up .venv ==="

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
CFLAGS="-std=c99" pip install soundfile pesq tqdm numpy einops librosa scipy joblib

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo "=== Done ==="
