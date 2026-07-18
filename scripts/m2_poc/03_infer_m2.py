#!/usr/bin/env python3
# scripts/m2_poc/03_infer_m2.py
#
# LLark M2 PoC — inference script.
# Given a trained checkpoint and an audio file, answers a question about the music.
#
# Usage:
#   python scripts/m2_poc/03_infer_m2.py \
#       --model-path checkpoints/llark-qwen2-poc \
#       --audio-file my_song.mp3 \
#       --prompt "What instruments do you hear?"
#
#   # Interactive mode (ask multiple questions):
#   python scripts/m2_poc/03_infer_m2.py \
#       --model-path checkpoints/llark-qwen2-poc \
#       --audio-file my_song.mp3 \
#       --interactive

import argparse
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── CLAP embedding ───────────────────────────────────────────────────────────

def extract_clap_embedding(audio_path: str, device: torch.device) -> torch.Tensor:
    """Extract a CLAP embedding from an audio file and return as a tensor."""
    try:
        import laion_clap
    except ImportError:
        raise ImportError("Run: pip install laion-clap")

    print("Loading CLAP model for audio encoding...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt()
    model.eval()
    model = model.to(device)

    print(f"Extracting CLAP embedding from: {audio_path}")
    with torch.no_grad():
        embedding = model.get_audio_embedding_from_filelist(
            x=[audio_path], use_tensor=True
        )
    # embedding shape: (1, 512) → (1, 1, 512) for num_frames=1
    embedding = embedding.unsqueeze(0).to(device)  # (1, 1, 512)
    return embedding


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device: torch.device):
    """Load the trained LLark Qwen2 model and tokenizer from a checkpoint."""
    from transformers import AutoTokenizer

    # Import the Qwen2 model so it registers with AutoModelForCausalLM
    from m2t.models.qwen2 import WrappedQwen2ForCausalLM, WrappedQwen2Config  # noqa: F401
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading model from: {model_path}")
    
    # 1. Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        print("[WARN] Tokenizer not found in checkpoint, loading from base Qwen/Qwen2-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", use_fast=True)
        # Manually register the custom audio tokens added during training
        from m2t.special_tokens import DEFAULT_AUDIO_PATCH_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN
        tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)

    # 2. Load base model config to find the backbone path
    # Typically, we load from "Qwen/Qwen2-0.5B-Instruct" directly
    base_model_path = "Qwen/Qwen2-0.5B-Instruct"

    print(f"Loading base model: {base_model_path}")
    # Load base config and force it to be our custom config class
    config = WrappedQwen2Config.from_pretrained(base_model_path)
    config.mm_hidden_size = 512
    
    model = WrappedQwen2ForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float32,  # MPS requires fp32
        device_map=None,
    )
    
    # Initialize adapter projector modules on the base model
    model.get_model().initialize_adapter_modules()

    # 3. Resize model token embeddings to match the tokenizer's vocabulary size (151649)
    # as saved in the PEFT checkpoints.
    model.resize_token_embeddings(len(tokenizer))

    # Initialize audio token IDs in AudioEncoderConfig
    model.initialize_audio_tokenizer(
        mm_use_audio_start_end=True,
        tokenizer=tokenizer,
        device=device,
    )

    # 4. Load the LoRA adapter weights
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        
        # Load non-lora trainables (the projector weights) if saved separately
        non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
        if os.path.exists(non_lora_path):
            print(f"Loading non-LoRA projector weights: {non_lora_path}")
            non_lora_state = torch.load(non_lora_path, map_location="cpu")
            # Strip model. prefix if present
            non_lora_state = {k.replace("base_model.model.", ""): v for k, v in non_lora_state.items()}
            model.load_state_dict(non_lora_state, strict=False)

    model = model.to(device)
    model.eval()

    return model, tokenizer


# ─── Inference ────────────────────────────────────────────────────────────────

def build_prompt(question: str, audio_first: bool = True) -> str:
    """Build the conversation prompt including the <audio> placeholder."""
    from m2t.special_tokens import DEFAULT_AUDIO_TOKEN
    from m2t.llava import conversation as conversation_lib

    if audio_first:
        user_turn = f"{DEFAULT_AUDIO_TOKEN}\n{question}"
    else:
        user_turn = f"{question}\n{DEFAULT_AUDIO_TOKEN}"

    # Format using the same conversation template as training
    header = f"{conversation_lib.default_conversation.system}\n\n"
    prompt = (
        header
        + f"### Human: {user_turn}\n"
        + "### Assistant:"
    )
    return prompt


def replace_audio_tokens(prompt: str, tokenizer, audio_token_len: int = 1) -> str:
    """Replace <audio> placeholder with the appropriate patch tokens."""
    from m2t.special_tokens import (
        DEFAULT_AUDIO_END_TOKEN,
        DEFAULT_AUDIO_PATCH_TOKEN,
        DEFAULT_AUDIO_START_TOKEN,
        DEFAULT_AUDIO_TOKEN,
    )

    replace_token = DEFAULT_AUDIO_START_TOKEN + (DEFAULT_AUDIO_PATCH_TOKEN * audio_token_len) + DEFAULT_AUDIO_END_TOKEN
    return prompt.replace(DEFAULT_AUDIO_TOKEN, replace_token)


@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    audio_embedding: torch.Tensor,
    question: str,
    max_new_tokens: int = 256,
    device: torch.device = None,
) -> str:
    """Run a single inference pass and return the generated text."""
    if device is None:
        device = next(model.parameters()).device

    # The CLAP embedding for a single clip has shape (1, 512).
    # When treated as a sequence of "frames", it's 1 frame of dim 512.
    # audio_token_len = number of patch tokens = number of frames.
    audio_token_len = audio_embedding.shape[1] if audio_embedding.ndim == 3 else 1

    prompt = build_prompt(question, audio_first=True)
    prompt = replace_audio_tokens(prompt, tokenizer, audio_token_len=audio_token_len)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # audio_embedding: (1, num_frames, dim) → pass as batch list
    # Model expects audio_encodings as [batch_size, num_frames, dim]
    audio_encodings = audio_embedding.to(device=device, dtype=torch.float32)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_encodings=audio_encodings,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLark M2 PoC — ask questions about music using a trained checkpoint."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained model checkpoint directory.",
    )
    parser.add_argument(
        "--audio-file",
        required=True,
        help="Path to an audio file (WAV, MP3, FLAC, etc.).",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this music.",
        help="Question to ask about the audio (ignored in --interactive mode).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode to ask multiple questions about the same clip.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"[ERROR] Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model checkpoint not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    # Extract CLAP embedding once (reused across questions in interactive mode)
    audio_embedding = extract_clap_embedding(args.audio_file, device)

    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    if args.interactive:
        print("\n=== LLark Interactive Mode ===")
        print(f"Audio: {args.audio_file}")
        print("Type your question and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if question.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if not question:
                continue
            response = run_inference(
                model, tokenizer, audio_embedding, question,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            print(f"LLark: {response}\n")
    else:
        print(f"\nQuestion: {args.prompt}")
        response = run_inference(
            model, tokenizer, audio_embedding, args.prompt,
            max_new_tokens=args.max_new_tokens, device=device,
        )
        print(f"Answer:   {response}")


if __name__ == "__main__":
    main()
