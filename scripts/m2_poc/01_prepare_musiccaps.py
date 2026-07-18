#!/usr/bin/env python3
# Copyright 2024 — LLark M2 PoC
#
# scripts/m2_poc/01_prepare_musiccaps.py
#
# Downloads MusicCaps from HuggingFace, fetches YouTube audio via yt-dlp,
# extracts CLAP embeddings, and saves a local JSONL file compatible with
# the LLark training pipeline (read_hf_dataset format).
#
# Usage:
#   python scripts/m2_poc/01_prepare_musiccaps.py \
#       --output data/musiccaps_train.jsonl \
#       --audio-dir data/musiccaps_audio \
#       --max-samples 5000
#
#   # Smoke test (10 samples):
#   python scripts/m2_poc/01_prepare_musiccaps.py \
#       --output data/musiccaps_smoke.jsonl \
#       --max-samples 10
#
# Output JSONL schema (one JSON object per line):
#   {
#     "id": "<youtube_id>",
#     "audio_encoding": [<flat list of 512 floats>],
#     "audio_encoding_shape": [1, 512],
#     "conversations": [
#       {"from": "human", "value": "<audio>\n<question>"},
#       {"from": "gpt",   "value": "<caption>"}
#     ]
#   }

import argparse
import json
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Prompt templates ────────────────────────────────────────────────────────
# We cycle through several question phrasings so the model learns to
# handle diverse queries, not just "describe this music".
CAPTION_PROMPTS = [
    "Describe this music.",
    "What do you hear in this audio clip?",
    "Give a detailed description of the music in this recording.",
    "Describe the sounds and musical characteristics of this clip.",
    "What kind of music is this? Describe it in detail.",
    "Listen to this audio and describe what you hear.",
]

# ─── CLAP setup ──────────────────────────────────────────────────────────────

def load_clap_model():
    """Load the LAION-CLAP model for audio embedding extraction."""
    try:
        import laion_clap
    except ImportError:
        raise ImportError(
            "laion_clap not installed. Run: pip install laion-clap"
        )

    log.info("Loading CLAP model (this downloads ~900MB on first run)...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt()  # downloads the default checkpoint automatically
    model.eval()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    log.info(f"CLAP running on: {device}")
    model = model.to(device)
    return model, device


def extract_clap_embedding(model, audio_path: str) -> Optional[np.ndarray]:
    """
    Extract a 512-dim CLAP embedding from an audio file.

    Returns:
        np.ndarray of shape (1, 512), or None on failure.
    """
    try:
        import laion_clap
        audio_embed = model.get_audio_embedding_from_filelist(
            x=[audio_path], use_tensor=False
        )
        # audio_embed shape: (1, 512)
        return audio_embed
    except Exception as e:
        log.warning(f"CLAP embedding failed for {audio_path}: {e}")
        return None


# ─── Audio download ───────────────────────────────────────────────────────────

def download_youtube_clip(
    ytid: str,
    start_s: float,
    end_s: float,
    output_dir: str,
) -> Optional[str]:
    """
    Download a 30-second clip from YouTube using yt-dlp.

    Returns:
        Path to the downloaded WAV file, or None on failure.
    """
    out_path = os.path.join(output_dir, f"{ytid}.wav")
    if os.path.exists(out_path):
        return out_path  # already cached

    url = f"https://www.youtube.com/watch?v={ytid}"
    duration = end_s - start_s

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"ffmpeg:-ss {start_s} -t {duration}",
        "--output", out_path,
        url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        return out_path
    except (subprocess.TimeoutExpired, Exception):
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare MusicCaps data for LLark M2 PoC training."
    )
    parser.add_argument(
        "--output",
        default="data/musiccaps_train.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--audio-dir",
        default="data/musiccaps_audio",
        help="Directory to cache downloaded audio files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="MusicCaps split to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Load MusicCaps ────────────────────────────────────────────────────────
    log.info("Loading MusicCaps from HuggingFace...")
    ds = load_dataset("google/MusicCaps", split=args.split)
    log.info(f"MusicCaps {args.split} split: {len(ds)} samples")

    if args.max_samples and args.max_samples < len(ds):
        indices = random.sample(range(len(ds)), args.max_samples)
        ds = ds.select(indices)
        log.info(f"Subsampled to {len(ds)} samples")

    # ── Load CLAP ─────────────────────────────────────────────────────────────
    clap_model, device = load_clap_model()

    # ── Process each sample ───────────────────────────────────────────────────
    output_path = Path(args.output)
    n_success = 0
    n_failed = 0

    with open(output_path, "w") as f_out:
        for sample in tqdm(ds, desc="Processing MusicCaps"):
            ytid = sample["ytid"]
            caption = sample["caption"]
            start_s = sample.get("start_s", 0.0)
            end_s = sample.get("end_s", 30.0)

            # 1. Download audio
            audio_path = download_youtube_clip(
                ytid=ytid,
                start_s=start_s,
                end_s=end_s,
                output_dir=args.audio_dir,
            )
            if audio_path is None:
                log.debug(f"Download failed: {ytid}")
                n_failed += 1
                continue

            # 2. Extract CLAP embedding (skip empty files — yt-dlp can produce 0-byte WAVs)
            if os.path.getsize(audio_path) < 4096:  # < 4KB is definitely not valid audio
                log.debug(f"Skipping empty/corrupt audio: {audio_path}")
                n_failed += 1
                continue

            embedding = extract_clap_embedding(clap_model, audio_path)
            if embedding is None:
                n_failed += 1
                continue

            # embedding shape: (1, 512)
            audio_encoding = embedding.flatten().tolist()
            audio_encoding_shape = [1, 512]

            # 3. Pick a random question prompt
            question = random.choice(CAPTION_PROMPTS)

            # 4. Randomly place <audio> before or after the question
            audio_first = random.random() > 0.5
            if audio_first:
                prompt_text = f"<audio>\n{question}"
            else:
                prompt_text = f"{question}\n<audio>"

            # 5. Build the conversation record
            record = {
                "id": ytid,
                "audio_encoding": audio_encoding,
                "audio_encoding_shape": audio_encoding_shape,
                "conversations": [
                    {"from": "human", "value": prompt_text},
                    {"from": "gpt", "value": caption},
                ],
            }

            f_out.write(json.dumps(record) + "\n")
            n_success += 1

    log.info(f"Done. Success: {n_success}, Failed/skipped: {n_failed}")
    log.info(f"Output written to: {output_path}")
    log.info(
        f"Yield rate: {n_success / (n_success + n_failed) * 100:.1f}%"
        if (n_success + n_failed) > 0
        else "No samples processed."
    )


if __name__ == "__main__":
    main()
