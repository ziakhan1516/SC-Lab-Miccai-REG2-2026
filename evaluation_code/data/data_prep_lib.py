"""Shared helpers for preparing ground truth and local test inputs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Callable

DATA_DIR = Path(__file__).resolve().parent
EVAL_ROOT = DATA_DIR.parent
CASES_DIR = DATA_DIR / "cases"
PREDICTIONS_DIR = DATA_DIR / "predictions"
GROUND_TRUTH_DIR = EVAL_ROOT / "ground_truth"
TEST_INPUT_DIR = EVAL_ROOT / "test" / "input"

INTERF0_CASES = CASES_DIR / "interf0"
INTERF1_CASES = CASES_DIR / "interf1"
INTERF0_ROIS_MAPPING = INTERF0_CASES / "rois_mapping.txt"
INTERF1_GT_COT = INTERF1_CASES / "ground_truth_CoT.json"

INTERF0_PREDICTIONS_FILE = PREDICTIONS_DIR / "interf0" / "predictions.json"
INTERF1_PREDICTIONS_FILE = PREDICTIONS_DIR / "interf1" / "predictions.json"

METRIC_A_GT = GROUND_TRUTH_DIR / "metric_A" / "chain-of-thoughts-ground-truth.json"
METRIC_B_MAPPING = GROUND_TRUTH_DIR / "metric_B" / "rois_mapping.txt"
MANIFEST = GROUND_TRUTH_DIR / "manifest.json"

ROIS_MAPPING_COLUMNS = [
    "anonymous_id",
    "image",
    "variant",
    "paired_anonymous_id",
    "b3_paired_anonymous_id",
    "label",
    "question",
]

ROIS_REQUIRED_COLUMNS = frozenset(ROIS_MAPPING_COLUMNS)

INTERF0_SOCKETS = (
    "histopathology-region-of-interest-thumbnail",
    "visual-context-question",
)
INTERF1_SOCKETS = ("whole-slide-image",)


class DataPrepError(ValueError):
    """Invalid or missing staged data for preparation scripts."""


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise DataPrepError(f"Expected file not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise DataPrepError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_pk(case_id: str) -> str:
    """
    Case key without file extension — used for manifest pk and test/input folders.

    Accepts bare ids (e.g. ``000000``) or filenames (e.g. ``PIT_01_00020_01.tiff``).
    """
    name = Path(str(case_id).strip()).name
    if not name:
        raise DataPrepError("Case id cannot be empty")
    return Path(name).stem if Path(name).suffix else name


def case_title_from_id(case_id: str) -> str:
    """Canonical case title for metric A (matches evaluate_metrics normalization)."""
    return normalize_pk(case_id)


def validate_rois_mapping(path: Path, *, allow_empty: bool = False) -> list[dict[str, str]]:
    if not path.is_file():
        raise DataPrepError(f"ROI mapping not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise DataPrepError(f"No header in {path}")
        missing = ROIS_REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise DataPrepError(
                f"{path} missing required columns: {sorted(missing)}"
            )
        for row_num, row in enumerate(reader, start=2):
            clean = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            if not clean.get("anonymous_id"):
                raise DataPrepError(f"{path} line {row_num}: missing anonymous_id")
            rows.append(clean)

    if not rows and not allow_empty:
        raise DataPrepError(f"{path}: no data rows (file is header-only or empty)")
    return rows


def write_empty_rois_mapping(path: Path) -> None:
    """Header-only metric B mapping when interf0 cases are not provided."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROIS_MAPPING_COLUMNS, delimiter="\t")
        writer.writeheader()


def normalize_metric_a_cases(raw: list[Any]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise DataPrepError("ground_truth_CoT.json must be a JSON array")
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, case in enumerate(raw):
        if not isinstance(case, dict):
            raise DataPrepError(f"Case {i} must be an object")
        case_id = case.get("id")
        cot = case.get("chain-of-thought")
        if not case_id:
            raise DataPrepError(f"Case {i} missing 'id'")
        if not isinstance(cot, list):
            raise DataPrepError(
                f"Case {case.get('id')!r} missing 'chain-of-thought' array"
            )
        norm_id = str(case_id)
        if norm_id in seen_ids:
            raise DataPrepError(f"Duplicate interf1 case id: {norm_id}")
        seen_ids.add(norm_id)
        out.append({"id": norm_id, "chain-of-thought": cot})
    return out


def _synthetic_manifest_value(*, pk: int, slug: str) -> dict[str, Any]:
    return {"pk": pk, "interface": {"slug": slug}}


def _interf0_manifest_values(*, upload_index: int) -> list[dict[str, Any]]:
    base_pk = 1_000_000 + upload_index * 100
    return [
        _synthetic_manifest_value(
            pk=base_pk, slug="visual-context-question"
        ),
        _synthetic_manifest_value(
            pk=base_pk + 1, slug="histopathology-region-of-interest-thumbnail"
        ),
    ]


def _interf1_manifest_values(*, upload_index: int) -> list[dict[str, Any]]:
    base_pk = 2_000_000 + upload_index * 100
    return [
        _synthetic_manifest_value(pk=base_pk, slug="whole-slide-image"),
    ]


def build_minimal_upload_manifest(
    *,
    interf0_rows: list[dict[str, str]],
    interf1_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Minimal manifest: uploads[].display_set.pk, title, and values[].

    Synthetic integer value pks and interface slugs allow evaluate.py to match
    anonymized prediction jobs back to case titles.
    """
    if not interf0_rows and not interf1_cases:
        raise DataPrepError(
            "Cannot build manifest.json with zero interf0 and interf1 cases"
        )

    uploads: list[dict[str, Any]] = []
    seen_pks: set[str] = set()

    for upload_index, row in enumerate(interf0_rows):
        title = normalize_pk(str(row["anonymous_id"]))
        if title in seen_pks:
            raise DataPrepError(f"Duplicate interf0 pk: {title}")
        seen_pks.add(title)
        uploads.append(
            {
                "case": {"title": title},
                "display_set": {
                    "pk": title,
                    "title": title,
                    "values": _interf0_manifest_values(upload_index=upload_index),
                },
            }
        )

    for upload_index, case in enumerate(interf1_cases):
        title = case_title_from_id(str(case["id"]))
        if title in seen_pks:
            raise DataPrepError(f"Duplicate interf1 pk: {title}")
        seen_pks.add(title)
        uploads.append(
            {
                "case": {"title": title},
                "display_set": {
                    "pk": title,
                    "title": title,
                    "values": _interf1_manifest_values(upload_index=upload_index),
                },
            }
        )

    return {"uploads": uploads}


def build_pk_to_title_from_manifest(data: dict[str, Any]) -> dict[str, str]:
    """
    Map lookup keys -> canonical case title.

    Supports minimal manifests (display_set.pk equals the case title) and Grand
    Challenge upload manifests (display_set.pk is a UUID). Lookup keys:

    - display_set.pk (platform job pk)
    - normalize_pk(case title) (prediction id, extension stripped)
    """
    uploads = data.get("uploads")
    if not isinstance(uploads, list):
        raise DataPrepError("manifest must contain an uploads array")
    if not uploads:
        raise DataPrepError("manifest has no upload entries")

    mapping: dict[str, str] = {}
    for entry in uploads:
        display_set = entry.get("display_set") or {}
        case = entry.get("case") or {}
        ds_pk = display_set.get("pk")
        title = display_set.get("title") or case.get("title")
        if not ds_pk or not title:
            raise DataPrepError(
                f"Invalid manifest entry (need display_set.pk and title): {entry}"
            )
        title_str = str(title)
        for key in (str(ds_pk), normalize_pk(title_str)):
            if key in mapping and mapping[key] != title_str:
                raise DataPrepError(
                    f"Conflicting manifest lookup keys for {key!r}: "
                    f"{mapping[key]!r} vs {title_str!r}"
                )
            mapping[key] = title_str
    return mapping


def build_case_key_to_job_pk_from_manifest(data: dict[str, Any]) -> dict[str, str]:
    """
    Map normalized case identifiers -> display_set.pk (test/input folder name).

    Accepts prediction ``id`` values with optional extensions (e.g. ``000000.jpg``)
    or bare anonymous_id / case title. Also accepts display_set.pk directly.
    """
    uploads = data.get("uploads")
    if not isinstance(uploads, list):
        raise DataPrepError("manifest must contain an uploads array")
    if not uploads:
        raise DataPrepError("manifest has no upload entries")

    mapping: dict[str, str] = {}
    for entry in uploads:
        display_set = entry.get("display_set") or {}
        case = entry.get("case") or {}
        ds_pk = display_set.get("pk")
        title = display_set.get("title") or case.get("title")
        if not ds_pk or not title:
            raise DataPrepError(
                f"Invalid manifest entry (need display_set.pk and title): {entry}"
            )
        job_pk = str(ds_pk)
        case_key = normalize_pk(str(title))
        for key in (case_key, job_pk):
            if key in mapping and mapping[key] != job_pk:
                raise DataPrepError(
                    f"Conflicting manifest case keys for {key!r}: "
                    f"{mapping[key]!r} vs {job_pk!r}"
                )
            mapping[key] = job_pk
    return mapping


def load_upload_manifest_pk_to_title(path: Path) -> dict[str, str]:
    return build_pk_to_title_from_manifest(load_json(path))


def load_manifest_case_key_to_job_pk(path: Path) -> dict[str, str]:
    return build_case_key_to_job_pk_from_manifest(load_json(path))


def load_optional_json_array(
    path: Path,
    *,
    label: str,
    on_load: Callable[[list[Any]], list[dict]],
) -> list[dict]:
    """Load a JSON array file, or return [] if the path does not exist."""
    if not path.exists():
        return []
    if not path.is_file():
        raise DataPrepError(f"{label} path is not a file: {path}")

    raw = load_json(path)
    if not isinstance(raw, list):
        raise DataPrepError(f"{label} ({path}) must be a JSON array")
    return on_load(raw)


def clear_test_input_dir() -> None:
    if TEST_INPUT_DIR.exists():
        for child in TEST_INPUT_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def _socket(slug: str, *, relative_path: str | None = None) -> dict[str, Any]:
    """Minimal socket stub — only fields read by evaluate.py."""
    sock: dict[str, Any] = {"slug": slug}
    if relative_path is not None:
        sock["relative_path"] = relative_path
    return {"socket": sock}


def find_upload_by_case_key(manifest: dict[str, Any], case_key: str) -> dict[str, Any]:
    """Return the single manifest upload whose case title matches ``case_key``."""
    uploads = manifest.get("uploads")
    if not isinstance(uploads, list):
        raise DataPrepError("manifest must contain an uploads array")

    matches: list[dict[str, Any]] = []
    for entry in uploads:
        display_set = entry.get("display_set") or {}
        case = entry.get("case") or {}
        title = display_set.get("title") or case.get("title")
        if title and normalize_pk(str(title)) == case_key:
            matches.append(entry)

    if not matches:
        raise DataPrepError(
            f"Case key {case_key!r} not found in manifest.json uploads"
        )
    if len(matches) > 1:
        raise DataPrepError(
            f"Case key {case_key!r} matches multiple manifest uploads"
        )
    return matches[0]


def build_minimal_job_from_upload(
    *,
    job_pk: str,
    upload: dict[str, Any],
    output_stubs: list[dict[str, Any]],
) -> dict[str, Any]:
    display_set = upload.get("display_set") or {}
    values = display_set.get("values")
    if not isinstance(values, list) or not values:
        case = upload.get("case") or {}
        title = display_set.get("title") or case.get("title")
        raise DataPrepError(
            f"Manifest entry for case {title!r} has no display_set.values[]"
        )

    inputs: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise DataPrepError("Each display_set.values entry must be an object")
        value_pk = value.get("pk")
        interface = value.get("interface") or {}
        slug = interface.get("slug")
        if value_pk is None or not slug:
            raise DataPrepError(
                "Each display_set.values entry must include pk and interface.slug"
            )
        inputs.append(
            {
                "pk": value_pk,
                "socket": {"slug": slug},
                "interface": {"slug": slug},
            }
        )

    return {
        "pk": job_pk,
        "url": f"https://local.test/jobs/{job_pk}/",
        "inputs": inputs,
        "outputs": output_stubs,
    }


def interf0_output_stubs() -> list[dict[str, Any]]:
    return make_interf0_job(pk="stub")["outputs"]


def interf1_output_stubs() -> list[dict[str, Any]]:
    return make_interf1_job(pk="stub")["outputs"]


def make_interf0_job(*, pk: str) -> dict[str, Any]:
    """
    Minimal predictions.json job for metric B.

    evaluate.py uses: pk; input slugs (interface detection); output slug + relative_path.
    Questions come from metric_B/rois_mapping.txt, not this stub.
    """
    return {
        "pk": pk,
        "inputs": [
            _socket("visual-context-question"),
            _socket("histopathology-region-of-interest-thumbnail"),
        ],
        "outputs": [
            _socket(
                "visual-context-response",
                relative_path="visual-context-response.json",
            ),
        ],
    }


def make_interf1_job(*, pk: str) -> dict[str, Any]:
    """
    Minimal predictions.json job for metric A.

    evaluate.py uses: pk; input slugs (interface detection); output slug + relative_path.
    """
    return {
        "pk": pk,
        "inputs": [_socket("whole-slide-image")],
        "outputs": [
            _socket("chain-of-thought", relative_path="chain-of-thought.json"),
        ],
    }
