"""Stage 2 of Visual Grounding (Metric B): ask the trained WSI reasoning model
an ROI-level question for every ROI and record its answer.

It reuses the *existing* model code in the parent Try2 folder (load_generator,
load_wsi_features) without modifying it, and the CONCH ROI features produced by
extract_roi_features.py. Each ROI feature is a [1, 512] bag (one patch), loaded
and L2-normalised exactly like the WSI patch features the model trained on.

The answers are written with the ROI metadata (tissue/background label,
original/perturbed variant, pairings) needed to compute B1/B2/B3 later.

Run AFTER the CONCH model is trained:
    python3 run_roi_grounding.py \
        --checkpoint /home/ali/storage1/Bin-Version2/Reg2/codings/Try2/checkpoints/wsi_reasoning_r1qwen1p5b_conch
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROI_DIR = HERE.parent                       # .../Try2/ROI
TRY2_DIR = ROI_DIR.parent                    # .../Try2 (has reasoning_mllm etc.)

# Reuse the trained-model code without copying/modifying it.
sys.path.insert(0, str(TRY2_DIR))
from multimodal_alignment import load_generator          # noqa: E402
from wsi_dataset import load_wsi_features                 # noqa: E402

DEFAULT_CKPT = TRY2_DIR / "checkpoints" / "wsi_reasoning_r1qwen1p5b_conch"

# Optional ROI-focused system prompt. The model was trained with a WSI-report
# system prompt that asks for a full <think> + pathology report; for ROI Q&A a
# short, grounded answer is wanted. Pass --roi-system-prompt to use this.
ROI_SYSTEM_PROMPT = (
    "You are an expert pathologist examining a single small region of interest "
    "(ROI) from a whole-slide image. Answer the question about THIS region only, "
    "briefly and based strictly on the visual evidence. If the region contains no "
    "meaningful tissue (e.g. background, blank, or non-informative area), say that "
    "it is background / not assessable and do not make any diagnostic, "
    "morphological, grading, or tumor-related claim."
)


def load_roi_metadata(mapping_path: Path) -> dict:
    """Parse anonymous_rois_mapping.txt (tab-separated) keyed by anonymous_id."""
    meta = {}
    if not mapping_path.exists():
        return meta
    with open(mapping_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            meta[row["anonymous_id"]] = {
                "label": row.get("label"),                # tissue | background
                "variant": row.get("variant"),            # original | perturbed
                "pair_id": row.get("pair_id"),
                "paired_anonymous_id": row.get("paired_anonymous_id"),
                "b3_paired_anonymous_id": row.get("b3_paired_anonymous_id"),
                "slide_id": row.get("slide_id"),
            }
    return meta


def main():
    ap = argparse.ArgumentParser(description="Ask the trained model ROI-level questions.")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="Trained CONCH reasoning checkpoint (feature_dim=512).")
    ap.add_argument("--features-dir", default=str(HERE / "roi_features"),
                    help="Per-ROI .pt features from extract_roi_features.py.")
    ap.add_argument("--pairs-json", default=str(ROI_DIR / "roi_question_pairs.json"))
    ap.add_argument("--mapping", default=str(ROI_DIR / "anonymous_rois_mapping.txt"))
    ap.add_argument("--output", default=str(HERE / "roi_grounding_answers.json"))
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--roi-system-prompt", action="store_true",
                    help="Override the model's WSI system prompt with an ROI-focused, "
                         "brief-answer prompt (recommended for grounding Q&A).")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not (ckpt / "model_config.json").exists():
        raise FileNotFoundError(
            f"No trained model at {ckpt}. Train the CONCH model first "
            f"(arch=reasoning, feature_dim=512), then rerun."
        )

    with open(args.pairs_json) as f:
        pairs = json.load(f)
    meta = load_roi_metadata(Path(args.mapping))

    print(f"Checkpoint  : {ckpt}")
    print(f"Features    : {args.features_dir}")
    print(f"ROIs        : {len(pairs)}")
    print(f"Device      : {device}\nLoading model ...")

    model = load_generator(str(ckpt), device=device)
    model.eval()
    if args.roi_system_prompt and hasattr(model, "system_prompt"):
        model.system_prompt = ROI_SYSTEM_PROMPT
        print("Using ROI-focused system prompt.")

    results = []
    for i, item in enumerate(pairs, 1):
        roi_id = item["id"]
        question = item.get("question", "")
        feat_path = Path(args.features_dir) / f"{roi_id}.pt"
        if not feat_path.exists():
            print(f"[{i}/{len(pairs)}] {roi_id}: missing feature -> skipped")
            continue

        # Same load path as WSI patches: raw [1,512] -> L2-normalised.
        features = load_wsi_features(str(feat_path))
        with torch.no_grad():
            answer = model.generate(
                features=[features],
                prompts=[question],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )[0].strip()

        rec = {
            "id": roi_id,
            "image": item.get("image"),
            "question": question,
            "answer": answer,
            **meta.get(roi_id, {}),
        }
        results.append(rec)
        lbl = rec.get("label", "?")
        print(f"\n[{i}/{len(pairs)}] {roi_id} ({lbl})\n  Q: {question}\n  A: {answer}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} ROI answers: {args.output}")
    print("Next: score B1 (background rejection), B2 (input sensitivity), "
          "B3 (cross-region consistency) with a pathology judge model over these answers.")


if __name__ == "__main__":
    main()
