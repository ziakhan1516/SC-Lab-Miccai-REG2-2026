#!/usr/bin/env python3
"""
Load a trained WSI model and generate responses, comparing with actual text.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from json_loader import load_json
from metrics import evaluate_generation
from multimodal_alignment import WSIReportGenerator
from text_preprocess import build_generation_prompt, preprocess_data
from wsi_dataset import discover_wsi_files, load_wsi_features, normalize_slide_id


def main():
    parser = argparse.ArgumentParser(
        description="Load model and generate responses with comparison"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--pt-dir",
        type=str,
        required=True,
        help="Directory containing WSI feature .pt files",
    )
    parser.add_argument(
        "--slide-id",
        type=str,
        required=True,
        help="Slide ID to generate response for",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Path to JSON file with reference texts (for comparison)",
    )
    parser.add_argument(
        "--root-key",
        type=str,
        default=None,
        help="Root key in JSON if data is nested",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Custom instruction/prompt prefix",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) probability for sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (cuda, cpu, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output JSON",
    )

    args = parser.parse_args()

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = WSIReportGenerator.from_pretrained(args.checkpoint, device=device)
    model.eval()
    print("✓ Model loaded")

    # Find WSI feature file
    print(f"\nSearching for slide: {args.slide_id}")
    file_map = discover_wsi_files(args.pt_dir)
    slide_key = normalize_slide_id(args.slide_id)

    if slide_key not in file_map:
        print(f"✗ No WSI feature file found for '{args.slide_id}'")
        print(f"Available slides: {list(file_map.keys())[:10]}...")
        return

    wsi_file = file_map[slide_key]
    print(f"✓ Found WSI feature file: {wsi_file}")

    # Load WSI features
    print(f"\nLoading WSI features...")
    features = load_wsi_features(str(wsi_file))
    print(f"✓ Loaded features with shape: {features.shape}")

    # Load reference text if JSON provided
    reference_text = None
    prompt = build_generation_prompt(args.instruction)

    if args.json:
        print(f"\nLoading reference data from: {args.json}")
        raw_data = load_json(args.json, root_key=args.root_key)
        processed_data = preprocess_data(raw_data)

        # Find matching sample
        for sample in processed_data:
            if normalize_slide_id(sample["slide_id"]) == slide_key:
                reference_text = sample["target_text"]
                prompt = sample["prompt"]
                print(f"✓ Found reference text for {slide_key}")
                break

        if reference_text is None:
            print(f"✗ No reference text found for {slide_key}")

    # Generate response
    print(f"\nGenerating response...")
    with torch.no_grad():
        generated = model.generate(
            features=[features],
            prompts=[prompt],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )[0].strip()

    print("✓ Generation complete")

    # Display results
    print("\n" + "=" * 80)
    print(f"SLIDE ID: {slide_key}")
    print("=" * 80)

    print("\n📝 PROMPT:")
    print("-" * 80)
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

    print("\n" + "=" * 80)
    print("🤖 GENERATED TEXT:")
    print("=" * 80)
    print(generated)

    metrics = None
    if reference_text:
        print("\n" + "=" * 80)
        print("📄 ACTUAL TEXT (Reference):")
        print("=" * 80)
        print(reference_text)

        # Calculate metrics
        metrics = evaluate_generation(generated, reference_text)

        # Show comparison length
        print("\n" + "=" * 80)
        print("📊 LENGTH COMPARISON:")
        print("=" * 80)
        print(f"Generated: {len(generated)} chars, {len(generated.split())} words")
        print(f"Reference: {len(reference_text)} chars, {len(reference_text.split())} words")

        # Show BLEU scores
        print("\n" + "=" * 80)
        print("🎯 BLEU SCORES:")
        print("=" * 80)
        bleu = metrics.get("bleu", {})
        for key, score in sorted(bleu.items()):
            print(f"  {key.upper()}: {score:.2f}")

        # Show ROUGE scores
        print("\n" + "=" * 80)
        print("🎯 ROUGE SCORES:")
        print("=" * 80)
        for rouge_type in ["rouge_1", "rouge_2", "rouge_l"]:
            if rouge_type in metrics:
                scores = metrics[rouge_type]
                print(f"\n  {rouge_type.upper()}:")
                print(f"    Precision: {scores['precision']:.2f}%")
                print(f"    Recall:    {scores['recall']:.2f}%")
                print(f"    F1:        {scores['f1']:.2f}%")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "slide_id": slide_key,
            "wsi_file": str(wsi_file),
            "prompt": prompt,
            "generated_text": generated,
            "reference_text": reference_text,
            "metrics": metrics,
            "config": {
                "checkpoint": args.checkpoint,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")

    # Final comparison section
    if reference_text:
        print("\n" + "=" * 80)
        print("📋 FINAL REPORT COMPARISON")
        print("=" * 80)
        
        print("\n" + "-" * 80)
        print("🤖 GENERATED REPORT:")
        print("-" * 80)
        print(generated)
        
        print("\n" + "-" * 80)
        print("📄 REFERENCE REPORT:")
        print("-" * 80)
        print(reference_text)
        
        print("\n" + "=" * 80)
        print("✓ Comparison Complete")
        print("=" * 80)


if __name__ == "__main__":
    main()
