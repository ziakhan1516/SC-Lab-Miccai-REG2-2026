"""Standalone Visual Grounding (Metric B) scorer for our ROI answers.

This does NOT re-implement anything: it imports the OFFICIAL challenge scorer
(Evaluation/REG2026/submission_evaluation_code/evaluate_metrics.py) and runs only
its visual-grounding half (B1/B2/B3) on our `roi_grounding_answers.json`, so the
number is exactly what the organisers would compute -- just isolated from the
Workflow-Reasoning (Metric A) half.

    Final Visual Score = 0.30 * B1 + 0.30 * B2 + 0.40 * B3
        B1 = Background Rejection      (background ROIs answered as non-informative)
        B2 = Input Sensitivity         (answer changes under patch-mask perturbation)
        B3 = Cross-region Consistency  (tissue vs background answers differ)

A local pathology *judge* LLM grades each answer (the official metric uses
Qwen3-8B). Qwen2.5-7B-Instruct works as a drop-in local judge; pass --judge-model-path
to the official Qwen3-8B for exact leaderboard parity.

Run with the `manga` env (loads torch 2.8 / transformers 4.57 + the judge):
    /home/ali/storage2/anaconda3/envs/manga/bin/python score_visual_grounding.py
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent           # .../Try2/ROI/visual_grounding
ROI_DIR = HERE.parent                             # .../Try2/ROI
TRY2_DIR = ROI_DIR.parent                         # .../Try2
EVAL_DIR = TRY2_DIR / "Evaluation" / "REG2026" / "submission_evaluation_code"

# Import the official scorer without copying/modifying it.
sys.path.insert(0, str(EVAL_DIR))
from evaluate_metrics import (                    # noqa: E402
    LocalQwenJudgeLLM,
    run_visual_dataset_from_answer_json,
    DEFAULT_W1,
    DEFAULT_W2,
    DEFAULT_W3,
    DEFAULT_VOTING,
)

# Official judge = Qwen3-8B. That snapshot is not downloaded here, but a complete
# Qwen2.5-7B-Instruct is, and it follows the same <answer>SAME/DIFFERENT</answer>
# protocol, so it is the default. Override with --judge-model-path for parity.
DEFAULT_JUDGE = (
    "/home/ali/storage4/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)


def main():
    ap = argparse.ArgumentParser(description="Score ONLY Visual Grounding (Metric B).")
    ap.add_argument("--visual-json", default=str(HERE / "roi_grounding_answers.json"),
                    help="Our ROI answers (from run_roi_grounding.py).")
    ap.add_argument("--mapping-txt", default=str(ROI_DIR / "anonymous_rois_mapping.txt"),
                    help="Official ROI mapping (tissue/background, pairings).")
    ap.add_argument("--judge-model-path", default=DEFAULT_JUDGE,
                    help="Local judge LLM. Official = Qwen3-8B; default here = local Qwen2.5-7B-Instruct.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="Judge generation budget. 512 suits a non-thinking instruct judge; "
                         "raise to ~32768 for a Qwen3 thinking judge.")
    ap.add_argument("--voting", type=int, default=DEFAULT_VOTING)
    ap.add_argument("--w1", type=float, default=DEFAULT_W1)
    ap.add_argument("--w2", type=float, default=DEFAULT_W2)
    ap.add_argument("--w3", type=float, default=DEFAULT_W3)
    ap.add_argument("--output", default=str(HERE / "visual_grounding_score.json"))
    args = ap.parse_args()

    if not Path(args.visual_json).exists():
        raise FileNotFoundError(
            f"{args.visual_json} not found. Run run_roi_grounding.py first."
        )

    print(f"Visual answers : {args.visual_json}")
    print(f"ROI mapping    : {args.mapping_txt}")
    print(f"Judge model    : {args.judge_model_path}")
    print(f"Weights        : w1={args.w1} w2={args.w2} w3={args.w3} | voting={args.voting}")
    print("Loading judge LLM ...\n")

    judge = LocalQwenJudgeLLM(
        model_path=args.judge_model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    summary = run_visual_dataset_from_answer_json(
        visual_answer_json_path=Path(args.visual_json),
        mapping_txt_path=Path(args.mapping_txt),
        judge_llm=judge,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        voting=args.voting,
    )

    print("\n" + "=" * 64)
    print("  VISUAL GROUNDING (Metric B) — official scoring")
    print("=" * 64)
    print(f"  B1  Background Rejection     : {summary['average_B1_background']:.4f}")
    print(f"  B2  Input Sensitivity        : {summary['average_B2_sensitivity']:.4f}")
    print(f"  B3  Cross-region Consistency : {summary['average_B3_cross_region']:.4f}")
    print("-" * 64)
    print(f"  FINAL VISUAL SCORE = {args.w1}*B1 + {args.w2}*B2 + {args.w3}*B3 "
          f"= {summary['final_visual_score']:.4f}")
    print("=" * 64)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full breakdown (per-ROI judgments) -> {args.output}")


if __name__ == "__main__":
    main()
