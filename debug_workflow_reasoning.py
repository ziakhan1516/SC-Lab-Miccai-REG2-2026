#!/usr/bin/env python3
"""
Debug script to inspect generated vs reference outputs and metric calculations.
"""

import argparse
import json
from pathlib import Path

import torch

from evaluate_workflow_reasoning import ChainOfThoughtParser, WorkflowReasoningMetrics, WorkflowReasoningMetrics
from json_loader import load_json
from multimodal_alignment import WSIReportGenerator
from text_preprocess import preprocess_data
from wsi_dataset import discover_wsi_files, load_wsi_features, normalize_slide_id


def main():
    parser = argparse.ArgumentParser(description="Debug workflow reasoning evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--root-key", default=None)
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to inspect")
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and model
    raw_data = load_json(args.json, root_key=args.root_key)
    processed_data = preprocess_data(raw_data)
    
    model = WSIReportGenerator.from_pretrained(args.checkpoint, device=device)
    model.eval()
    
    file_map = discover_wsi_files(args.pt_dir)
    parser_cot = ChainOfThoughtParser()
    metrics_computer = WorkflowReasoningMetrics()

    print("=" * 100)
    print("DEBUGGING WORKFLOW REASONING EVALUATION")
    print("=" * 100)

    for idx, sample in enumerate(processed_data[:args.num_samples], 1):
        slide_id = normalize_slide_id(sample["slide_id"])
        
        if slide_id not in file_map:
            print(f"\n[{idx}] ⚠️ {slide_id} - No WSI file")
            continue

        try:
            # Generate response
            features = load_wsi_features(str(file_map[slide_id]))
            generated = model.generate(
                features=[features],
                prompts=[sample["prompt"]],
                max_new_tokens=1000,
                do_sample=False,
            )[0].strip()

            # Parse CoT
            pred_cot = parser_cot.parse_cot(generated)
            gt_cot = parser_cot.parse_cot(sample["target_text"])
            
            # Extract edges
            pred_edges = parser_cot.extract_edges(pred_cot)
            gt_edges = parser_cot.extract_edges(gt_cot)

            print(f"\n{'='*100}")
            print(f"SAMPLE {idx}: {slide_id}")
            print(f"{'='*100}")

            print(f"\n📝 REFERENCE TEXT (first 300 chars):")
            print(f"{sample['target_text'][:300]}...")
            
            print(f"\n🤖 GENERATED TEXT (first 300 chars):")
            print(f"{generated[:300]}...")

            print(f"\n🔗 GROUND TRUTH EDGES ({len(gt_edges)} edges):")
            for i, edge in enumerate(sorted(gt_edges), 1):
                print(f"  {i}. {edge[0][:40]}... -> {edge[1][:40]}...")

            print(f"\n🔗 PREDICTED EDGES ({len(pred_edges)} edges):")
            for i, edge in enumerate(sorted(pred_edges), 1):
                print(f"  {i}. {edge[0][:40]}... -> {edge[1][:40]}...")

            print(f"\n📊 GT CoT STEPS ({len(gt_cot)} steps):")
            for i, step in enumerate(gt_cot[:3], 1):
                print(f"  {i}. Q: {step.get('question', 'N/A')[:60]}...")
                print(f"     A: {step.get('answer', 'N/A')[:60]}...")

            print(f"\n📊 PRED CoT STEPS ({len(pred_cot)} steps):")
            for i, step in enumerate(pred_cot[:3], 1):
                print(f"  {i}. Q: {step.get('question', 'N/A')[:60]}...")
                print(f"     A: {step.get('answer', 'N/A')[:60]}...")

            # Metrics
            bpv = metrics_computer.binary_path_validity(pred_edges, gt_edges)
            edge_f1 = metrics_computer.edge_f1(pred_edges, gt_edges)
            mess = metrics_computer.mean_edge_semantic_similarity(
                pred_cot, gt_cot, pred_edges, gt_edges
            )
            
            print(f"\n✅ METRICS:")
            print(f"  BPV:      {bpv:.4f} {'✓ MATCH' if bpv == 1.0 else '✗ MISMATCH'}")
            print(f"  Edge-F1:  {edge_f1:.4f} {'✓ PERFECT' if edge_f1 == 1.0 else '✗ PARTIAL'}")
            print(f"  MESS:     {mess:.4f}")

        except Exception as e:
            print(f"\n[{idx}] ✗ Error: {e}")

    print(f"\n{'='*100}")
    print("DEBUG COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
