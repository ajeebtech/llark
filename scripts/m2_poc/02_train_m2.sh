#!/usr/bin/env bash
# scripts/m2_poc/02_train_m2.sh
#
# LLark M2 PoC training script — single-process, MPS-accelerated.
# Trains a Qwen2-0.5B-based LLark model with LoRA + CLAP embeddings.
#
# Prerequisites:
#   1. conda activate llark-m2
#   2. python scripts/m2_poc/01_prepare_musiccaps.py (generate training data)
#
# Usage:
#   bash scripts/m2_poc/02_train_m2.sh
#
# For a quick smoke test (5 steps only):
#   MAX_STEPS=5 bash scripts/m2_poc/02_train_m2.sh

set -e

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen2-0.5B-Instruct"
TRAIN_DATA="data/musiccaps_train.jsonl"
OUTPUT_DIR="checkpoints/llark-qwen2-poc"
MAX_STEPS="${MAX_STEPS:-3000}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== LLark M2 PoC Training ==="
echo "  Model:      $MODEL"
echo "  Data:       $TRAIN_DATA"
echo "  Output:     $OUTPUT_DIR"
echo "  Max steps:  $MAX_STEPS"
echo ""

if [ ! -f "$TRAIN_DATA" ]; then
  echo "[ERROR] Training data not found: $TRAIN_DATA"
  echo "  Run first: python scripts/m2_poc/01_prepare_musiccaps.py"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Check MPS is available
python -c "
import torch
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[INFO] Training device: {device}')
if device == 'cpu':
    print('[WARN] MPS not available — training will be slow on CPU.')
"

# Single-process training (no torch.distributed — single M2 GPU)
python -m m2t.train \
  --model_name_or_path "$MODEL" \
  --version v0 \
  --output_dir "$OUTPUT_DIR" \
  --mm_hidden_size 512 \
  --mm_use_audio_start_end \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --freeze_backbone False \
  --tune_mm_mlp_adapter True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_steps "$MAX_STEPS" \
  --model_max_length 1024 \
  --bf16 False \
  --fp16 False \
  --tf32 False \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --dataloader_num_workers 0 \
  --remove_unused_columns False \
  --eval_strategy no \
  --report_to none \
  --train_data_path "$TRAIN_DATA"

echo ""
echo "=== Training complete! ==="
echo "  Checkpoint saved to: $OUTPUT_DIR"
echo "  Run inference with:"
echo "    python scripts/m2_poc/03_infer_m2.py --model-path $OUTPUT_DIR --audio-file your_song.mp3"
