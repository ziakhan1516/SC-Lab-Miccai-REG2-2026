import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from json_loader import load_json
from metrics import evaluate_generation
from multimodal_alignment import WSIReportGenerator
from text_preprocess import build_generation_prompt, preprocess_data
from wsi_dataset import load_wsi_features, normalize_slide_id


DEFAULT_CHECKPOINT = (
    "/home/ali/storage1/Bin-Version2/Reg2/codings/Try1/outputs/wsi_sft/epoch_3"
)
DEFAULT_JSON = "/home/ali/storage4/Bin/reg2026/dataset Reg/train_CoT.json"


def _processed_by_slide_id(json_path: str, root_key: Optional[str]) -> Dict[str, dict]:
    processed = preprocess_data(load_json(json_path, root_key=root_key))
    return {
        normalize_slide_id(sample["slide_id"]): sample
        for sample in processed
    }


def _format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    return json.dumps(metrics, indent=2)


def run_manual_check(args) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pt_path = Path(args.pt_file)
    slide_id = normalize_slide_id(args.slide_id or pt_path.stem)

    references = {}
    if args.json:
        references = _processed_by_slide_id(args.json, args.root_key)

    reference = references.get(slide_id)
    prompt = reference["prompt"] if reference else build_generation_prompt(args.instruction)
    reference_text = reference["target_text"] if reference else ""

    print(f"Checkpoint: {args.checkpoint}")
    print(f"PT file: {pt_path}")
    print(f"Slide ID: {slide_id}")
    print(f"Device: {device}")
    print(f"Max new tokens: {args.max_new_tokens}")

    model = WSIReportGenerator.from_pretrained(args.checkpoint, device=device)
    features = load_wsi_features(str(pt_path))

    generated_text = model.generate(
        features=[features],
        prompts=[prompt],
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )[0].strip()

    metrics = evaluate_generation(generated_text, reference_text) if reference_text else {}

    result = {
        "slide_id": slide_id,
        "pt_file": str(pt_path),
        "checkpoint": args.checkpoint,
        "generated_text": generated_text,
        "actual_text": reference_text,
        "metrics": metrics,
    }

    print("\n================ GENERATED ================\n")
    print(generated_text)

    if reference_text:
        print("\n================ ACTUAL / REFERENCE ================\n")
        print(reference_text)
        print("\n================ SCORES ================\n")
        print(_format_metrics(metrics))
    else:
        print("\nNo matching reference text was found, so scores were not computed.")
        print("Pass --json and make sure the .pt filename stem matches slide_id/id.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved manual check output: {output_path}")

    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Manually check one WSI .pt file against a trained SFT checkpoint. "
            "Prints generated text, actual/reference text, BLEU, and ROUGE."
        )
    )
    parser.add_argument("--pt-file", required=True, help="Path to one .pt/.pth WSI feature file.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json", default=DEFAULT_JSON, help="Reference CoT/report JSON.")
    parser.add_argument("--root-key", default=None)
    parser.add_argument(
        "--slide-id",
        default=None,
        help="Optional slide ID override. Defaults to the .pt filename stem.",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=950)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_manual_check(args)


if __name__ == "__main__":
    main()
