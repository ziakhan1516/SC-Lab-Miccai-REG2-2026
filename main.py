import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import torch

from build_wsi_embeddings import WSIEmbeddingPipeline
from json_loader import load_json, write_jsonl
from metrics import average_metrics, evaluate_generation
from multimodal_alignment import WSIReportGenerator, load_generator
from text_preprocess import build_generation_prompt, preprocess_data
from train import WSISFTTrainer
from wsi_dataset import discover_wsi_files, load_wsi_features, normalize_slide_id


def load_processed(json_path: str, root_key: str = None):
    raw = load_json(json_path, root_key=root_key)
    processed = preprocess_data(raw)
    if not processed:
        raise ValueError("No valid supervised samples were produced from the JSON.")
    return processed


def _is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _write_json(records, file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _matched_samples(processed, pt_dir: str):
    file_map = discover_wsi_files(pt_dir)
    matched = []
    missing = []

    for sample in processed:
        slide_id = normalize_slide_id(sample["slide_id"])
        if slide_id in file_map:
            matched.append(sample)
        else:
            missing.append(sample)

    return matched, missing


def _train_test_split(samples, test_size: float, seed: int):
    if not 0.0 <= test_size < 1.0:
        raise ValueError("--test-size must be in [0.0, 1.0).")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    if test_size == 0.0:
        return shuffled, []

    test_count = max(1, int(round(len(shuffled) * test_size)))
    test_data = shuffled[:test_count]
    train_data = shuffled[test_count:]
    return train_data, test_data


def _prepare_training_split(processed, args):
    if args.test_size <= 0:
        return processed, []

    matched, missing = _matched_samples(processed, args.pt_dir)
    train_data, test_data = _train_test_split(
        matched,
        test_size=args.test_size,
        seed=args.seed,
    )

    split_dir = Path(args.split_output_dir or Path(args.output_dir) / "splits")
    if _is_main_rank():
        _write_json(train_data, str(split_dir / "train.json"))
        _write_json(test_data, str(split_dir / "test.json"))
        _write_json(
            {
                "seed": args.seed,
                "test_size": args.test_size,
                "all_processed_samples": len(processed),
                "matched_samples": len(matched),
                "missing_wsi_samples": len(missing),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
            },
            str(split_dir / "metadata.json"),
        )
        print(
            f"Saved split: train={len(train_data)} test={len(test_data)} "
            f"missing_wsi={len(missing)} -> {split_dir}"
        )

    return train_data, test_data


def _parse_max_memory(spec):
    """'0=10GiB,1=22GiB' -> {0: '10GiB', 1: '22GiB'}."""
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        key, value = part.split("=", 1)
        key = key.strip()
        out[int(key) if key.isdigit() else key] = value.strip()
    return out


def _select_eval_samples(test_data, num_samples, seed):
    if not test_data or num_samples <= 0:
        return []
    shuffled = list(test_data)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:num_samples]


def cmd_preprocess(args):
    processed = load_processed(args.json, root_key=args.root_key)
    write_jsonl(processed, args.output)
    print(f"Processed samples: {len(processed)}")


def cmd_inspect(args):
    processed = load_processed(args.json, root_key=args.root_key)
    wsi_files = discover_wsi_files(args.pt_dir)
    matched = [
        sample
        for sample in processed
        if normalize_slide_id(sample["slide_id"]) in wsi_files
    ]

    print(f"Processed samples: {len(processed)}")
    print(f"WSI feature files: {len(wsi_files)}")
    print(f"Matched samples: {len(matched)}")

    first = matched[0] if matched else processed[0]
    print("\nFirst sample:")
    print(f"  id: {first['id']}")
    print(f"  slide_id: {first['slide_id']}")
    print("\nPrompt preview:")
    print(first["prompt"][:800])
    print("\nTarget preview:")
    print(first["target_text"][:1200])


def cmd_train(args):
    processed = load_processed(args.json, root_key=args.root_key)
    train_data, test_data = _prepare_training_split(processed, args)

    eval_samples = _select_eval_samples(
        test_data,
        num_samples=args.eval_num_samples,
        seed=args.eval_seed if args.eval_seed is not None else args.seed,
    )

    trainer = WSISFTTrainer(
        processed_data=train_data,
        pt_dir=args.pt_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        arch=args.arch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        feature_dim=args.feature_dim,
        mil_hidden_dim=args.mil_hidden_dim,
        attention_dim=args.attention_dim,
        prefix_length=args.prefix_length,
        num_visual_tokens=args.num_visual_tokens,
        resampler_depth=args.resampler_depth,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        think_format=not args.disable_think,
        load_in_4bit=args.load_in_4bit,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_target_length,
        freeze_language_model=args.freeze_language_model,
        missing_wsi=args.missing_wsi,
        device=args.device,
        device_map=args.device_map,
        max_memory=_parse_max_memory(args.max_memory),
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        static_graph=args.static_graph,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_samples=eval_samples,
        eval_max_new_tokens=args.eval_max_new_tokens or args.max_target_length,
        eval_use_embeddings=args.eval_embeddings,
        eval_embedding_model=args.eval_embedding_model,
    )
    trainer.train()


def _resolve_infer_files(args) -> List[Path]:
    if args.pt_file:
        return [Path(args.pt_file)]

    if not args.pt_dir:
        raise ValueError("Provide --pt-file or --pt-dir for inference.")

    file_map = discover_wsi_files(args.pt_dir)
    if args.slide_id:
        key = normalize_slide_id(args.slide_id)
        if key not in file_map:
            raise FileNotFoundError(f"No WSI feature file found for '{key}'.")
        return [file_map[key]]

    return list(file_map.values())


def cmd_infer(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(args.checkpoint, device=device)
    prompt = build_generation_prompt(args.instruction)
    files = _resolve_infer_files(args)

    results = []
    for file_path in files:
        features = load_wsi_features(str(file_path))
        outputs = model.generate(
            features=[features],
            prompts=[prompt],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        result = {
            "slide_id": file_path.stem,
            "file_path": str(file_path),
            "generated_text": outputs[0].strip(),
        }
        results.append(result)

        print(f"\n=== {file_path.stem} ===")
        print(result["generated_text"])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved inference output: {output_path}")


def cmd_evaluate(args):
    processed = load_processed(args.json, root_key=args.root_key)
    file_map = discover_wsi_files(args.pt_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(args.checkpoint, device=device)

    samples = []
    for sample in processed:
        slide_id = normalize_slide_id(sample["slide_id"])
        if args.slide_id and slide_id != normalize_slide_id(args.slide_id):
            continue
        if slide_id not in file_map:
            continue
        samples.append(sample)
        if len(samples) >= args.limit:
            break

    if not samples:
        raise ValueError("No matched evaluation samples were found.")

    results = []
    metric_rows = []

    for sample in samples:
        slide_id = normalize_slide_id(sample["slide_id"])
        features = load_wsi_features(str(file_map[slide_id]))
        generated = model.generate(
            features=[features],
            prompts=[sample["prompt"]],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )[0].strip()

        reference = sample["target_text"]
        metrics = evaluate_generation(generated, reference)
        metric_rows.append(metrics)

        result = {
            "id": sample["id"],
            "slide_id": slide_id,
            "file_path": str(file_map[slide_id]),
            "generated_text": generated,
            "reference_text": reference,
            "metrics": metrics,
        }
        results.append(result)

        print(f"\n=== {slide_id} ===")
        print("\nGenerated:")
        print(generated)
        print("\nReference:")
        print(reference[:args.reference_preview_chars])
        print("\nMetrics:")
        print(json.dumps(metrics, indent=2))

    summary = average_metrics(metric_rows)
    payload = {
        "checkpoint": args.checkpoint,
        "num_samples": len(results),
        "summary": summary,
        "results": results,
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation output: {output_path}")


def cmd_embeddings(args):
    pipeline = WSIEmbeddingPipeline(
        pt_dir=args.pt_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint,
        device=args.device,
    )
    embeddings = pipeline.generate_embeddings()
    pipeline.save_embeddings(embeddings, args.output)


def build_parser():
    parser = argparse.ArgumentParser(
        description="WSI supervised fine-tuning for pathology reasoning/report generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--json", required=True)
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--root-key", default=None)
    preprocess.set_defaults(func=cmd_preprocess)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--json", required=True)
    inspect.add_argument("--pt-dir", required=True)
    inspect.add_argument("--root-key", default=None)
    inspect.set_defaults(func=cmd_inspect)

    train = subparsers.add_parser("train")
    train.add_argument("--json", required=True)
    train.add_argument("--pt-dir", required=True)
    train.add_argument("--output-dir", default="checkpoints/wsi_sft")
    train.add_argument("--root-key", default=None)
    train.add_argument(
        "--arch",
        choices=["causal", "seq2seq", "reasoning"],
        default="reasoning",
        help="reasoning = WSI-conditioned reasoning LLM (Perceiver resampler + "
        "DeepSeek-R1-Distill/LoRA, true generation with <think>); seq2seq = BioBART "
        "encoder-decoder; causal = decoder-only soft-prefix LM.",
    )
    train.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    train.add_argument(
        "--num-visual-tokens",
        type=int,
        default=576,
        help="WSI visual tokens (L_img) the resampler emits / decoder attends to "
        "(576 = 24x24 grid; use 1024 for finer spatial granularity).",
    )
    train.add_argument(
        "--disable-think",
        action="store_true",
        help="Train without the <think>...</think> wrapper (plain Reasoning:/Report: target).",
    )
    train.add_argument(
        "--resampler-depth",
        type=int,
        default=3,
        help="Number of cross/self-attention layers in the Perceiver resampler (reasoning).",
    )
    train.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="QLoRA: load the frozen base LLM in 4-bit (NF4) so a 7B fits a "
        "single 24 GB GPU. LoRA adapters still train in bf16 (reasoning arch).",
    )
    train.add_argument("--lora-r", type=int, default=16, help="LoRA rank (reasoning).")
    train.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (reasoning).")
    train.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (reasoning).")
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument(
        "--feature-dim",
        type=int,
        default=512,
        help="Patch feature dimension. CONCH=512, UNI/ResNet-IN=1024. Must match "
        "the .pt files in --pt-dir; saved to model_config.json for reload.",
    )
    train.add_argument("--mil-hidden-dim", type=int, default=512)
    train.add_argument("--attention-dim", type=int, default=128)
    train.add_argument("--prefix-length", type=int, default=8)
    train.add_argument("--max-prompt-length", type=int, default=512)
    train.add_argument("--max-target-length", type=int, default=2048)
    train.add_argument("--test-size", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--split-output-dir", default=None)
    train.add_argument("--freeze-language-model", action="store_true")
    train.add_argument("--missing-wsi", choices=["error", "skip"], default="error")
    train.add_argument("--ddp-find-unused-parameters", action="store_true")
    train.add_argument(
        "--static-graph",
        action="store_true",
        help="DDP static_graph mode — required for multi-GPU + --gradient-checkpointing "
        "(ignored on single GPU).",
    )
    train.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for much lower activation memory (fixes OOM on long sequences).",
    )
    train.add_argument("--device", default=None)
    train.add_argument(
        "--device-map",
        default=None,
        help="Shard the LLM across the visible GPUs (e.g. 'auto') for model-parallel "
        "training of 8B/14B. Run with plain `python` (NOT torchrun).",
    )
    train.add_argument(
        "--max-memory",
        default=None,
        help="Per-GPU memory cap for --device-map, e.g. '0=10GiB,1=22GiB'. "
        "Set this when a GPU is already partly occupied.",
    )
    # In-training Workflow Reasoning evaluation (runs on rank 0 after each epoch).
    train.add_argument(
        "--eval-num-samples",
        type=int,
        default=10,
        help="Random held-out test cases to generate+score after each epoch (0 disables).",
    )
    train.add_argument("--eval-seed", type=int, default=None)
    train.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens for in-training eval generation (default: --max-target-length).",
    )
    train.add_argument(
        "--eval-embeddings",
        action="store_true",
        help="Use PubMedBERT embeddings for in-training MESS/Final cosine "
        "(default: fast lexical fallback).",
    )
    train.add_argument(
        "--eval-embedding-model",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    train.set_defaults(func=cmd_train)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--pt-file", default=None)
    infer.add_argument("--pt-dir", default=None)
    infer.add_argument("--slide-id", default=None)
    infer.add_argument("--instruction", default=None)
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.add_argument("--do-sample", action="store_true")
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--top-p", type=float, default=0.9)
    infer.add_argument("--output", default=None)
    infer.add_argument("--device", default=None)
    infer.set_defaults(func=cmd_infer)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--json", required=True)
    evaluate.add_argument("--pt-dir", required=True)
    evaluate.add_argument("--root-key", default=None)
    evaluate.add_argument("--slide-id", default=None)
    evaluate.add_argument("--limit", type=int, default=1)
    evaluate.add_argument("--max-new-tokens", type=int, default=700)
    evaluate.add_argument("--do-sample", action="store_true")
    evaluate.add_argument("--temperature", type=float, default=0.7)
    evaluate.add_argument("--top-p", type=float, default=0.9)
    evaluate.add_argument("--reference-preview-chars", type=int, default=1200)
    evaluate.add_argument("--output", default=None)
    evaluate.add_argument("--device", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    embeddings = subparsers.add_parser("embeddings")
    embeddings.add_argument("--pt-dir", required=True)
    embeddings.add_argument("--checkpoint", default=None)
    embeddings.add_argument("--output", default="wsi_slide_embeddings.pt")
    embeddings.add_argument("--batch-size", type=int, default=1)
    embeddings.add_argument("--num-workers", type=int, default=0)
    embeddings.add_argument("--device", default=None)
    embeddings.set_defaults(func=cmd_embeddings)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
