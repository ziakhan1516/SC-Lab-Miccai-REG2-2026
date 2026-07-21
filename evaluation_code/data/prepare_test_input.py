#!/usr/bin/env python3
"""
Build test/input/ (evaluation container layout) from data/predictions/.

Expected inputs (at least one required):
  data/predictions/interf0/predictions.json   — [{ "id": "<pk>", "answer": "..." }, ...]
  data/predictions/interf1/predictions.json   — [{ "id": "<pk>", "chain-of-thought": [...] }, ...]

Requires ground_truth/manifest.json (run prepare_test_ground_truth.sh first).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from data_prep_lib import (
    DataPrepError,
    INTERF0_PREDICTIONS_FILE,
    INTERF1_PREDICTIONS_FILE,
    MANIFEST,
    METRIC_B_MAPPING,
    TEST_INPUT_DIR,
    build_minimal_job_from_upload,
    clear_test_input_dir,
    find_upload_by_case_key,
    interf0_output_stubs,
    interf1_output_stubs,
    load_json,
    load_optional_json_array,
    normalize_pk,
    validate_rois_mapping,
    write_json,
)


def _parse_interf0_entries(raw: list) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DataPrepError(f"interf0/predictions.json[{i}] must be an object")
        case_id = item.get("id")
        if not case_id:
            raise DataPrepError(f"interf0/predictions.json[{i}] missing 'id'")
        if "answer" not in item:
            raise DataPrepError(f"interf0/predictions.json[{i}] missing 'answer'")
        source_id = str(case_id)
        pk = normalize_pk(source_id)
        if pk in seen:
            raise DataPrepError(f"Duplicate interf0 prediction id: {source_id!r}")
        seen.add(pk)
        out.append(
            {"case_key": pk, "source_id": source_id, "answer": str(item["answer"])}
        )
    return out


def _parse_interf1_entries(raw: list) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DataPrepError(f"interf1/predictions.json[{i}] must be an object")
        case_id = item.get("id")
        if not case_id:
            raise DataPrepError(f"interf1/predictions.json[{i}] missing 'id'")
        cot = item.get("chain-of-thought")
        if not isinstance(cot, list):
            raise DataPrepError(
                f"interf1/predictions.json[{i}] missing 'chain-of-thought' array"
            )
        source_id = str(case_id)
        pk = normalize_pk(source_id)
        if pk in seen:
            raise DataPrepError(f"Duplicate interf1 prediction id: {source_id!r}")
        seen.add(pk)
        out.append(
            {
                "case_key": pk,
                "source_id": source_id,
                "chain-of-thought": cot,
            }
        )
    return out


def write_visual_response(path: Path, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False)
        f.write("\n")


def write_chain_of_thought(path: Path, steps: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare test/input from data/predictions JSON files."
    )
    parser.add_argument(
        "--interf0",
        type=Path,
        default=INTERF0_PREDICTIONS_FILE,
        help="interf0 predictions JSON (optional if missing)",
    )
    parser.add_argument(
        "--interf1",
        type=Path,
        default=INTERF1_PREDICTIONS_FILE,
        help="interf1 predictions JSON (optional if missing)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="manifest.json (from prepare_test_ground_truth.sh)",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise DataPrepError(
            f"Missing {args.manifest}. Run ./prepare_test_ground_truth.sh first."
        )

    interf0_preds = load_optional_json_array(
        args.interf0,
        label="interf0/predictions.json",
        on_load=_parse_interf0_entries,
    )
    interf1_preds = load_optional_json_array(
        args.interf1,
        label="interf1/predictions.json",
        on_load=_parse_interf1_entries,
    )

    if not interf0_preds and not interf1_preds:
        raise DataPrepError(
            "No predictions to export. Provide at least one of:\n"
            f"  {args.interf0}\n"
            f"  {args.interf1}\n"
            "Each file may be absent or a JSON array (empty array is allowed)."
        )

    manifest = load_json(args.manifest)

    questions_by_id: dict[str, str] = {}
    if interf0_preds:
        if not METRIC_B_MAPPING.is_file():
            raise DataPrepError(
                f"Missing {METRIC_B_MAPPING}. "
                "Run ./prepare_test_ground_truth.sh with interf0 cases first."
            )
        questions_by_id = {
            normalize_pk(row["anonymous_id"]): row["question"]
            for row in validate_rois_mapping(METRIC_B_MAPPING, allow_empty=True)
        }
        if not questions_by_id:
            raise DataPrepError(
                f"{METRIC_B_MAPPING} has no ROI rows but interf0 predictions were given"
            )

    for pred in interf0_preds:
        case_key = pred["case_key"]
        pred["upload"] = find_upload_by_case_key(manifest, case_key)
        if case_key not in questions_by_id:
            raise DataPrepError(
                f"interf0 id {pred['source_id']!r} (case key {case_key!r}) not in "
                f"{METRIC_B_MAPPING.name}"
            )

    for pred in interf1_preds:
        pred["upload"] = find_upload_by_case_key(manifest, pred["case_key"])

    clear_test_input_dir()
    jobs: list[dict] = []
    seen_job_pks: set[str] = set()

    for pred in interf0_preds:
        job_pk = pred["case_key"]
        if job_pk in seen_job_pks:
            raise DataPrepError(f"Duplicate generated interf0 job pk: {job_pk!r}")
        seen_job_pks.add(job_pk)
        jobs.append(
            build_minimal_job_from_upload(
                job_pk=job_pk,
                upload=pred["upload"],
                output_stubs=interf0_output_stubs(),
            )
        )
        out_path = (
            TEST_INPUT_DIR / job_pk / "output" / "visual-context-response.json"
        )
        write_visual_response(out_path, pred["answer"])

    for pred in interf1_preds:
        job_pk = pred["case_key"]
        if job_pk in seen_job_pks:
            raise DataPrepError(f"Duplicate generated interf1 job pk: {job_pk!r}")
        seen_job_pks.add(job_pk)
        jobs.append(
            build_minimal_job_from_upload(
                job_pk=job_pk,
                upload=pred["upload"],
                output_stubs=interf1_output_stubs(),
            )
        )
        out_path = TEST_INPUT_DIR / job_pk / "output" / "chain-of-thought.json"
        write_chain_of_thought(out_path, pred["chain-of-thought"])

    write_json(TEST_INPUT_DIR / "predictions.json", jobs)

    parts = []
    if interf0_preds:
        parts.append(f"{len(interf0_preds)} interf0")
    if interf1_preds:
        parts.append(f"{len(interf1_preds)} interf1")
    print(
        f"Wrote {TEST_INPUT_DIR}/predictions.json ({' + '.join(parts)} job(s))"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataPrepError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
