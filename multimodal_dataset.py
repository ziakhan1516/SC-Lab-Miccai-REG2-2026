from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset

from wsi_dataset import discover_wsi_files, load_wsi_features, normalize_slide_id


class MultiModalWSIDataset(Dataset):
    """
    Paired SFT dataset.

    Each item contains:
      - WSI patch features from a .pt/.pth file
      - a prompt describing the task
      - target text with Reasoning + Pathology Report
    """

    def __init__(
        self,
        json_data: List[Dict[str, Any]],
        pt_dir: str,
        target_field: str = "target_text",
        missing: str = "error",
        normalize_features: bool = True,
    ):
        if missing not in {"error", "skip"}:
            raise ValueError("missing must be 'error' or 'skip'.")

        self.target_field = target_field
        self.normalize_features = normalize_features
        self.file_map = discover_wsi_files(pt_dir)
        self.items: List[Dict[str, Any]] = []
        missing_slide_ids = []

        for record in json_data:
            slide_id = normalize_slide_id(record.get("slide_id", record.get("id", "")))
            if not slide_id:
                continue

            file_path = self.file_map.get(slide_id)
            if file_path is None:
                missing_slide_ids.append(slide_id)
                if missing == "skip":
                    continue
                continue

            target_text = record.get(target_field) or record.get("training_text")
            if not target_text:
                continue

            self.items.append(
                {
                    "id": record.get("id", slide_id),
                    "slide_id": slide_id,
                    "file_path": file_path,
                    "prompt": record.get("prompt", ""),
                    "target_text": target_text,
                    "reasoning": record.get("reasoning", ""),
                    "pathology_report": record.get("pathology_report", ""),
                }
            )

        if missing_slide_ids and missing == "error":
            preview = ", ".join(missing_slide_ids[:10])
            raise FileNotFoundError(
                f"Missing {len(missing_slide_ids)} WSI feature files in {pt_dir}. "
                f"First missing IDs: {preview}"
            )

        if not self.items:
            raise ValueError("No usable WSI/text SFT samples were found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        item["features"] = load_wsi_features(
            str(item["file_path"]),
            normalize=self.normalize_features,
        )
        item["file_path"] = str(Path(item["file_path"]))
        return item


def collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "slide_id": [item["slide_id"] for item in batch],
        "file_path": [item["file_path"] for item in batch],
        "features": [item["features"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "reasoning": [item["reasoning"] for item in batch],
        "pathology_report": [item["pathology_report"] for item in batch],
    }
