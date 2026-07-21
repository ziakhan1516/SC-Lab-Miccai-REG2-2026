"""
Interface 0 — Visual Grounding (Metric B).

ROI thumbnail + question
  -> CONCH ViT-B-16 feature  [1, 512]
  -> WSI reasoning generation model (conditioned on the ROI feature)
  -> short, visually-grounded answer string.

CONCH is used ONLY for feature extraction; the answer is produced by the
trained generation model. A region with no tissue should be answered as
background / not assessable (the ROI system prompt steers this).
"""

from __future__ import annotations

from pathlib import Path

from core import load_json_file, load_roi_image

# Vendored on PYTHONPATH=/opt/app/lib inside the container.
from engine import get_engine, strip_think

# ROI-focused prompt: brief, grounded, reject background. Replaces the heavy
# WSI-report system prompt for single-ROI questions.
ROI_SYSTEM_PROMPT = (
    "You are an expert pathologist examining a single small region of interest "
    "(ROI) from a whole-slide image. Answer the question about THIS region only, "
    "briefly and based strictly on the visual evidence. If the region contains no "
    "meaningful tissue (e.g. background, blank, or non-informative area), say it is "
    "background / not assessable and do not make any diagnostic, morphological, "
    "grading, or tumor-related claim."
)

MAX_NEW_TOKENS = 128


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    question = load_json_file(location=question_path)
    if isinstance(question, dict):
        question = str(question.get("question", question))
    else:
        question = str(question)

    roi_image = load_roi_image(location=roi_image_path)

    conch, generator, _ = get_engine()
    features = conch.encode_roi(roi_image)            # [1, 512], L2-normalised

    # Use the ROI-focused system prompt for this turn.
    saved_prompt = getattr(generator, "system_prompt", None)
    if hasattr(generator, "system_prompt"):
        generator.system_prompt = ROI_SYSTEM_PROMPT
    try:
        raw = generator.generate(
            features=[features],
            prompts=[question],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )[0]
    finally:
        if saved_prompt is not None:
            generator.system_prompt = saved_prompt

    answer = strip_think(raw)
    return answer or raw.strip()
