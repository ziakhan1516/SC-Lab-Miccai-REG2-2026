"""Load the trained WSI reasoning generation model **fully offline**.

The container has no internet, but reasoning_mllm builds its base LLM by HF name
(deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). So the base LLM is bundled in the
model tarball as `<MODEL_PATH>/base_llm/`, and this loader overrides `lm_name`
in the saved config to that local directory before constructing the model.

Layout under /opt/ml/model (produced by prepare_model_dir.py):

    base_llm/                     full DeepSeek base weights + tokenizer (offline)
    generation/                   the trained checkpoint
        model_config.json         (lm_name may be the HF name; overridden here)
        lora_adapter/             trained LoRA adapter
        resampler.pt              trained Perceiver resampler
        tokenizer/                trained tokenizer
    conch/pytorch_model.bin       CONCH ViT-B-16 weights
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch

# Hard offline — never reach the network at runtime.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_generation_model(
    model_root: Path,
    device: str = "cpu",
    generation_subdir: str = "generation",
    base_llm_subdir: str = "base_llm",
):
    """Return a ready-to-generate WSIReasoningReportGenerator on `device`."""
    from reasoning_mllm import WSIReasoningReportGenerator
    from peft import PeftModel
    from transformers import AutoTokenizer

    gen_dir = Path(model_root) / generation_subdir
    base_dir = Path(model_root) / base_llm_subdir

    with (gen_dir / "model_config.json").open() as f:
        config = json.load(f)
    config.pop("arch", None)

    # Point the base LLM at the bundled local copy (offline).
    if base_dir.exists():
        config["lm_name"] = str(base_dir)

    # Build the model skeleton (no PEFT yet), then attach the trained adapter.
    model = WSIReasoningReportGenerator(_build_peft=False, **config)
    model.llm = PeftModel.from_pretrained(model.llm, str(gen_dir / "lora_adapter"))

    # Fold the LoRA deltas into the base weights. For a non-quantized (bf16) base
    # this is mathematically identical to the wrapped-adapter forward, but removes
    # the per-layer LoRA matmuls at every decode step -> faster generation with
    # NO change to outputs. Skipped automatically if the base were 4-bit.
    if not config.get("load_in_4bit", False):
        try:
            model.llm = model.llm.merge_and_unload()
            print("[gen] LoRA adapter merged into base weights (faster decode).")
        except Exception as e:  # never let an optimisation break model loading
            print(f"[gen] merge_and_unload skipped ({e}); using wrapped adapter.")

    model.resampler.load_state_dict(
        torch.load(gen_dir / "resampler.pt", map_location="cpu")
    )
    if (gen_dir / "tokenizer").exists():
        model.tokenizer = AutoTokenizer.from_pretrained(
            str(gen_dir / "tokenizer"), use_fast=True
        )

    model = model.to(device)
    model.eval()
    return model
