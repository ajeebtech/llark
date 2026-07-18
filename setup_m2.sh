#!/usr/bin/env bash
# setup_m2.sh — One-shot environment setup for LLark M2 PoC
#
# Usage:
#   chmod +x setup_m2.sh && ./setup_m2.sh
#
# This creates a conda env named 'llark-m2' with all M2-compatible deps.

set -e

ENV_NAME="llark-m2"
PYTHON_VERSION="3.11"

echo "==> Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "==> Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Installing PyTorch with MPS support"
# Install PyTorch — the standard pip version includes MPS on macOS
pip install torch>=2.2.0 torchaudio>=2.2.0

echo "==> Installing remaining requirements"
pip install -r requirements-m2.txt

echo "==> Verifying MPS availability"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if not torch.backends.mps.is_available():
    print('[WARN] MPS not available — will fall back to CPU. Training will be slow.')
else:
    print('[OK] MPS is available. Training will use Apple Silicon GPU.')
"

echo "==> Verifying CLAP"
python -c "import laion_clap; print('[OK] laion_clap imported successfully')"

echo "==> Verifying transformers / Qwen2"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('[OK] transformers imported successfully')
# Check Qwen2 is available
from transformers import Qwen2Config
print('[OK] Qwen2Config available')
"

echo ""
echo "======================================"
echo " Setup complete!"
echo " Activate with: conda activate $ENV_NAME"
echo " Next step: python scripts/m2_poc/01_prepare_musiccaps.py --help"
echo "======================================"
