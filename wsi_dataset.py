from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


FEATURE_KEYS = ("features", "patch_features", "h", "x", "embeddings")
WSI_SUFFIXES = (".pt", ".pth")


def discover_wsi_files(pt_dir: str) -> Dict[str, Path]:
    root = Path(pt_dir)
    files: Dict[str, Path] = {}

    for suffix in WSI_SUFFIXES:
        for file_path in sorted(root.glob(f"*{suffix}")):
            files[file_path.stem] = file_path

    return files


def normalize_slide_id(slide_id: str) -> str:
    return Path(str(slide_id)).stem


def _torch_load(file_path: Path):
    try:
        return torch.load(file_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(file_path, map_location="cpu")
    except Exception:
        return torch.load(file_path, map_location="cpu", weights_only=False)


def _extract_feature_tensor(data, file_path: Path) -> torch.Tensor:
    if isinstance(data, dict):
        feats = None
        for key in FEATURE_KEYS:
            if key in data:
                feats = data[key]
                break
        if feats is None:
            available = ", ".join(sorted(map(str, data.keys())))
            raise ValueError(
                f"{file_path} does not contain any feature key from "
                f"{FEATURE_KEYS}. Available keys: {available}"
            )
    else:
        feats = data

    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)

    feats = feats.float()

    if feats.dim() == 1:
        feats = feats.unsqueeze(0)

    if feats.dim() != 2:
        raise ValueError(
            f"Expected WSI features shaped [num_patches, feature_dim], "
            f"got {tuple(feats.shape)} from {file_path}"
        )

    return feats


def load_wsi_features(file_path: str, normalize: bool = True) -> torch.Tensor:
    path = Path(file_path)
    data = _torch_load(path)
    feats = _extract_feature_tensor(data, path)

    if normalize:
        feats = F.normalize(feats, p=2, dim=-1)

    return feats


class WSIDataset(Dataset):
    def __init__(
        self,
        pt_dir: str,
        slide_ids: Optional[Iterable[str]] = None,
        normalize: bool = True,
    ):
        self.normalize = normalize
        file_map = discover_wsi_files(pt_dir)

        if slide_ids is None:
            self.items: List[Tuple[str, Path]] = sorted(file_map.items())
        else:
            self.items = []
            for slide_id in slide_ids:
                key = normalize_slide_id(slide_id)
                if key not in file_map:
                    raise FileNotFoundError(f"No WSI feature file found for '{key}'.")
                self.items.append((key, file_map[key]))

        print(f"Found {len(self.items)} WSI files")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        slide_id, file_path = self.items[idx]
        feats = load_wsi_features(str(file_path), normalize=self.normalize)
        return feats, slide_id


def collate_fn(batch):
    feats, wsi_ids = zip(*batch)
    return list(feats), list(wsi_ids)
