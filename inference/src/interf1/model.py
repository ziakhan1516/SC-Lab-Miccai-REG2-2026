"""
Interface 1 — Workflow Reasoning (Metric A).

WSI (.tiff)
  -> CONCH ViT-B-16 patch features  [N, 512]  (pyramid check + CLAM seg/patch)
  -> WSI reasoning generation model (DeepSeek-R1-Distill-Qwen + resampler + LoRA)
  -> generated <think> Question/Answer/Next Question ... </think> Pathology Report
  -> parsed chain-of-thought (question / answer / next_question).

CONCH is used ONLY for feature extraction; the reasoning chain is produced by
the trained generation model. Non-pyramidal slides are converted in-process
(pyvips) before patching, matching the training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

# Vendored on PYTHONPATH=/opt/app/lib inside the container.
from engine import get_engine
from cot_parser import parse_cot_with_report
from text_preprocess import build_generation_prompt


class ChainOfThoughtStep(TypedDict):
    question: str
    answer: str
    next_question: str


# The reasoning chain can be long; allow enough tokens for the full workflow.
MAX_NEW_TOKENS = 2048


def predict_chain_of_thought(*, wsi_path: Path) -> "list[ChainOfThoughtStep]":
    conch, generator, _ = get_engine()

    # WSI -> [N, 512] CONCH bag (L2-normalised, like training).
    features = conch.extract_wsi(str(wsi_path))
    if features.shape[0] == 0:
        # No tissue patches found (e.g. blank/background-only slide). A fake
        # zero-vector "patch" would be out-of-distribution for the resampler,
        # so fail this prediction cleanly — the caller falls back to a single
        # "not assessable" step rather than crashing the whole case.
        raise RuntimeError(f"No tissue patches extracted from {wsi_path}")

    raw = generator.generate(
        features=[features],
        prompts=[build_generation_prompt()],   # same instruction used in training
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )[0]

    # Steps including the canonical 'What is the final pathology report?' step
    # (its answer is the report, which the evaluator scores).
    steps_raw = parse_cot_with_report(raw)

    steps: "list[ChainOfThoughtStep]" = []
    for s in steps_raw:
        steps.append({
            "question": str(s.get("question", "")),
            "answer": str(s.get("answer", "")),
            "next_question": str(s.get("next_question", "")),
        })
    if steps:
        steps[-1]["next_question"] = ""
    else:
        # Never emit an empty file; provide a minimal valid single step.
        steps = [{
            "question": "What type of specimen is this?",
            "answer": raw.strip()[:512] or "Not assessable from the provided slide.",
            "next_question": "",
        }]
    return steps
