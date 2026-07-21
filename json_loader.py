import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


JsonRecord = Dict[str, Any]


def _unwrap_records(data: Any, root_key: Optional[str] = None) -> Any:
    if root_key:
        if not isinstance(data, dict) or root_key not in data:
            raise ValueError(f"JSON root key '{root_key}' was not found.")
        return data[root_key]

    if isinstance(data, dict):
        for candidate in ("samples", "data", "records", "items"):
            if isinstance(data.get(candidate), list):
                return data[candidate]

    return data


def load_json(file_path: str, root_key: Optional[str] = None) -> List[JsonRecord]:
    """
    Load a pathology training JSON file.

    Accepted roots:
      - a list of sample dictionaries
      - a dictionary containing samples/data/records/items
      - a dictionary containing a user-specified root_key
    """
    path = Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format in {path}: {exc}") from exc

    records = _unwrap_records(data, root_key=root_key)

    if not isinstance(records, list):
        raise ValueError(
            "JSON root must be a list, or a dictionary with one of: "
            "samples, data, records, items."
        )

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {idx} is not a JSON object.")

    logging.info("Loaded %s samples from %s", len(records), path)
    return records


def write_jsonl(records: Iterable[JsonRecord], file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    logging.info("Wrote JSONL: %s", path)
