"""Load a trained SFT model and score it on the held-out test split.

This reuses exactly the pieces the in-training eval uses (`load_generator`,
`WorkflowReasoningMetrics`, `ground_truth_from_sample`) so the numbers are
directly comparable to the per-epoch `eval_epoch_*.json` reports.

Example
-------
python evaluate_test_set.py \
    --checkpoint /home/ali/storage1/Bin-Version2/Reg2/codings/Try1/checkpoints/wsi_reasoning_r1qwen1p5b

By default it reads the test set saved next to the checkpoint
(``<checkpoint>/splits/test.json``) and the WSI feature dir recorded in
``<checkpoint>/training_args.json``. Both can be overridden on the CLI.
"""

import argparse
import json
from pathlib import Path

import torch

from multimodal_alignment import load_generator
from wsi_dataset import load_wsi_features, discover_wsi_files, normalize_slide_id
from evaluate_workflow_reasoning import (
    WorkflowReasoningMetrics,
    TextEmbedder,
    ground_truth_from_sample,
)


def _resolve_paths(args):
    ckpt = Path(args.checkpoint)

    test_json = Path(args.test_json) if args.test_json else ckpt / "splits" / "test.json"
    if not test_json.exists():
        raise FileNotFoundError(
            f"Test split not found: {test_json}. Pass --test-json explicitly."
        )

    pt_dir = args.pt_dir
    if pt_dir is None:
        targs = ckpt / "training_args.json"
        if targs.exists():
            with targs.open() as f:
                pt_dir = json.load(f).get("pt_dir")
    if not pt_dir:
        raise ValueError("Could not resolve --pt-dir (no training_args.json). Pass it.")

    return test_json, pt_dir


def main():
    parser = argparse.ArgumentParser(description="Score a trained SFT model on the test split.")
    parser.add_argument(
        "--checkpoint",
        default="/home/ali/storage1/Bin-Version2/Reg2/codings/Try1/checkpoints/wsi_reasoning_r1qwen1p5b",
        help="Directory of the saved SFT model.",
    )
    parser.add_argument("--test-json", default=None, help="Override path to test split json.")
    parser.add_argument("--pt-dir", default=None, help="Override WSI .pt feature directory.")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (auto if omitted).")
    parser.add_argument("--limit", type=int, default=0, help="Only score first N test cases (0 = all).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Use PubMedBERT embeddings for MESS/Final cosine (default: fast lexical fallback).",
    )
    parser.add_argument(
        "--embedding-model",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the full JSON report (default: <checkpoint>/test_eval.json).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print generated vs reference text per case.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    test_json, pt_dir = _resolve_paths(args)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test split : {test_json}")
    print(f"WSI pt dir : {pt_dir}")
    print(f"Device     : {device}\n")

    # --- Load test samples and WSI feature file map -------------------------
    with open(test_json) as f:
        test_data = json.load(f)
    file_map = discover_wsi_files(pt_dir)

    # --- Load the trained model --------------------------------------------
    print("Loading model ...")
    model = load_generator(args.checkpoint, device=device)
    model.eval()

    # --- Scorer (same metric as in-training eval) --------------------------
    embedder = TextEmbedder(
        model_name=args.embedding_model,
        device=device,
        disabled=not args.use_embeddings,
    )
    scorer = WorkflowReasoningMetrics(embedder)

    results = []
    skipped = 0
    total = len(test_data) if args.limit <= 0 else min(args.limit, len(test_data))

    print(f"Scoring {total} test cases ...\n")
    for idx, sample in enumerate(test_data, start=1):
        if args.limit and idx > args.limit:
            break

        slide_id = normalize_slide_id(sample["slide_id"])
        file_path = file_map.get(slide_id)
        if file_path is None:
            skipped += 1
            print(f"[{idx}/{total}] {slide_id}: no WSI feature file -> skipped")
            continue

        features = load_wsi_features(str(file_path))
        with torch.no_grad():
            generated = model.generate(
                features=[features],
                prompts=[sample["prompt"]],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )[0].strip()

        gt_steps, gt_report = ground_truth_from_sample(sample)
        m = scorer.score_case(
            generated,
            sample["target_text"],
            gt_steps=gt_steps,
            gt_report=gt_report,
        )
        results.append(
            {
                "id": sample.get("id"),
                "slide_id": slide_id,
                "generated_text": generated,
                "reference_text": sample["target_text"],
                "metrics": m,
            }
        )

        print(
            f"[{idx}/{total}] {slide_id}  "
            f"BPV={m['binary_path_validity']:.3f} "
            f"EdgeF1={m['edge_f1']['f1']:.3f} "
            f"MESS={m['mess']:.3f} "
            f"Final={m['final_report_score']['score']:.3f} "
            f"WR={m['workflow_reasoning_score']:.3f}"
        )
        if args.verbose:
            print(f"{'-'*80}\n--- GENERATED ---\n{generated}")
            print(f"\n--- REFERENCE ---\n{sample['target_text']}\n{'-'*80}")

    if not results:
        raise SystemExit("No test cases were scored (all skipped / empty split).")

    # --- Aggregate ----------------------------------------------------------
    def _avg(key_fn):
        return sum(key_fn(r["metrics"]) for r in results) / len(results)

    summary = {
        "binary_path_validity": _avg(lambda m: m["binary_path_validity"]),
        "edge_f1": _avg(lambda m: m["edge_f1"]["f1"]),
        "mess": _avg(lambda m: m["mess"]),
        "final_report_score": _avg(lambda m: m["final_report_score"]["score"]),
        "workflow_reasoning_score": _avg(lambda m: m["workflow_reasoning_score"]),
    }

    print(f"\n{'='*80}")
    print(f"TEST SET SCORE  ({len(results)} cases scored, {skipped} skipped)")
    print(f"{'='*80}")
    print(json.dumps(summary, indent=2))
    print(
        f"\n>>> Overall Workflow Reasoning score on test set: "
        f"{summary['workflow_reasoning_score']:.4f}"
    )

    out_path = Path(args.output) if args.output else Path(args.checkpoint) / "test_eval.json"
    payload = {
        "checkpoint": args.checkpoint,
        "test_json": str(test_json),
        "num_scored": len(results),
        "num_skipped": skipped,
        "embeddings": args.embedding_model if args.use_embeddings else "lexical-fallback",
        "summary": summary,
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full report: {out_path}")


if __name__ == "__main__":
    main()
