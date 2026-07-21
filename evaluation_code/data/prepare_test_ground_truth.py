#!/usr/bin/env python3
"""
Build ground_truth/ from data/cases/.

At least one interface must be provided:
  data/cases/interf0/rois_mapping.txt     (metric B, optional)
  data/cases/interf1/ground_truth_CoT.json (metric A, optional)

Writes:
  ground_truth/metric_B/rois_mapping.txt
  ground_truth/metric_A/chain-of-thoughts-ground-truth.json
  ground_truth/manifest.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from data_prep_lib import (
    DataPrepError,
    INTERF0_ROIS_MAPPING,
    INTERF1_GT_COT,
    MANIFEST,
    METRIC_A_GT,
    METRIC_B_MAPPING,
    build_minimal_upload_manifest,
    load_json,
    normalize_metric_a_cases,
    validate_rois_mapping,
    write_empty_rois_mapping,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare test ground_truth from data/cases (interf0 and/or interf1)."
    )
    parser.add_argument(
        "--interf0-mapping",
        type=Path,
        default=INTERF0_ROIS_MAPPING,
        help="Source rois_mapping.txt for metric B (optional)",
    )
    parser.add_argument(
        "--interf1-cot",
        type=Path,
        default=INTERF1_GT_COT,
        help="Source ground_truth_CoT.json for metric A (optional)",
    )
    args = parser.parse_args()

    has_interf0 = args.interf0_mapping.is_file()
    has_interf1 = args.interf1_cot.is_file()

    if args.interf0_mapping.exists() and not has_interf0:
        raise DataPrepError(f"Not a file: {args.interf0_mapping}")
    if args.interf1_cot.exists() and not has_interf1:
        raise DataPrepError(f"Not a file: {args.interf1_cot}")

    if not has_interf0 and not has_interf1:
        raise DataPrepError(
            "No case data found. Provide at least one of:\n"
            f"  interf0: {args.interf0_mapping}\n"
            f"  interf1: {args.interf1_cot}"
        )

    interf0_rows: list[dict[str, str]] = []
    if has_interf0:
        interf0_rows = validate_rois_mapping(
            args.interf0_mapping, allow_empty=True
        )
        print(f"interf0: {len(interf0_rows)} ROI row(s) from {args.interf0_mapping}")
    else:
        print(f"interf0: skipped (no {args.interf0_mapping})")

    interf1_cases: list[dict] = []
    if has_interf1:
        interf1_cases = normalize_metric_a_cases(load_json(args.interf1_cot))
        print(f"interf1: {len(interf1_cases)} WSI case(s) from {args.interf1_cot}")
    else:
        print(f"interf1: skipped (no {args.interf1_cot})")

    METRIC_B_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    METRIC_A_GT.parent.mkdir(parents=True, exist_ok=True)

    if has_interf0:
        shutil.copy2(args.interf0_mapping, METRIC_B_MAPPING)
    else:
        write_empty_rois_mapping(METRIC_B_MAPPING)

    write_json(METRIC_A_GT, interf1_cases)

    manifest = build_minimal_upload_manifest(
        interf0_rows=interf0_rows,
        interf1_cases=interf1_cases,
    )
    write_json(MANIFEST, manifest)

    print(f"Wrote {METRIC_B_MAPPING}")
    print(f"Wrote {METRIC_A_GT} ({len(interf1_cases)} case(s))")
    print(f"Wrote {MANIFEST} ({len(manifest['uploads'])} manifest entries)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataPrepError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
