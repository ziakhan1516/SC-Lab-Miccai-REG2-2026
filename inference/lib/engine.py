"""Shared inference engine: CONCH encoder + WSI-reasoning generation model.

Loaded once per container (lazy singleton) and used by both interfaces. Weights
come from /opt/ml/model (overridable via REG_MODEL_DIR for local testing).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from conch_pipeline import CONCHEncoder
from generation_model import load_generation_model

_ENGINE = None  # (conch, generator, device)


def _model_dir() -> Path:
    # MODEL_PATH = /opt/ml/model on the platform; REG_MODEL_DIR for local tests.
    from core import MODEL_PATH
    return Path(os.environ.get("REG_MODEL_DIR", str(MODEL_PATH)))


def get_engine():
    """Return (conch_encoder, generator, device); build on first call."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = _model_dir()

    conch = CONCHEncoder(
        conch_ckpt_path=str(mdir / "conch" / "pytorch_model.bin"),
        device=device,
    )
    generator = load_generation_model(mdir, device=device)

    _ENGINE = (conch, generator, device)
    return _ENGINE


def strip_think(text: str) -> str:
    """Return the answer after a <think>...</think> block (the model's final
    answer); fall back to the whole text if there is no think wrapper."""
    text = str(text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "").strip()
    # Drop a leading 'Pathology Report:' / 'Answer:' label if present.
    for label in ("Pathology Report:", "Answer:", "Response:"):
        if text.lower().startswith(label.lower()):
            text = text[len(label):].strip()
    return text.strip()
