"""Real CLAM WSI pipeline, embedded for in-container inference.

This reproduces `wsi_full_pipeline.py` exactly, in-process, so the patch bag the
slot classifier sees inside the container is the SAME bag it was trained on:

    raw <uid>.tiff
      -> pyramid check (pyvips); if not pyramidal, tiffsave a pyramidal copy
      -> CLAM tissue segmentation (WholeSlideImage.segmentTissue)
      -> CLAM coordinate patching (process_contours) -> coords .h5
      -> openslide read_region per coord -> timm resnet50.tv_in1k (layer3 + pool)
      -> [N, 1024] bag -> L2-normalize each row (load_wsi_features normalize=True)

Segmentation / patching params match the values CLAM auto-recorded for the
training run (process_list_autogen.csv): seg_level/vis_level auto-selected via
get_best_level_for_downsample(64); sthresh=8, mthresh=7, close=4, use_otsu=False;
filter a_t=15, a_h=4, max_n_holes=16; patch_size=step_size=256 @ patch_level 0;
use_padding=True, contour_fn='four_pt'.

The timm encoder is built WITHOUT a pretrained download (offline) and the bundled
resnet50.pth (standard torchvision ImageNet weights) is loaded into it — verified
to satisfy every layer3-truncated key with zero missing keys.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

# CLAM modules are vendored under lib/clam and placed on PYTHONPATH so these
# absolute imports resolve exactly as in the original repo.
from wsi_core.WholeSlideImage import WholeSlideImage
from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from utils.transform_utils import get_eval_transforms
from utils.constants import IMAGENET_MEAN, IMAGENET_STD

try:
    import pyvips
    PYVIPS_OK = True
except Exception:
    PYVIPS_OK = False


# CLAM params recorded for the training run (process_list_autogen.csv).
SEG_PARAMS = dict(sthresh=8, mthresh=7, close=4, use_otsu=False)
FILTER_PARAMS = dict(a_t=15, a_h=4, max_n_holes=16)
PATCH_PARAMS = dict(use_padding=True, contour_fn="four_pt")
PATCH_SIZE = 256
STEP_SIZE = 256
PATCH_LEVEL = 0
JPEG_Q = 90


def _is_pyramidal(slide_path: str) -> bool:
    """A slide is usable by the CLAM pipeline if OpenSlide exposes >1 level.

    OpenSlide is the source of truth here (the whole pipeline reads regions
    through it), so detect pyramidalness with OpenSlide rather than pyvips —
    pyvips is only needed to *create* a pyramid when one is missing.
    """
    try:
        import openslide
        return openslide.open_slide(slide_path).level_count > 1
    except Exception:
        return False


def _ensure_pyramidal(slide_path: str, temp_root: str) -> "tuple[str, str | None]":
    """Return (path_to_use, temp_dir_to_cleanup_or_None). Mirrors ensure_pyramidal."""
    if _is_pyramidal(slide_path):
        return slide_path, None
    if not PYVIPS_OK:
        raise RuntimeError("Slide not pyramidal and pyvips unavailable.")
    os.makedirs(temp_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="pyr_", dir=temp_root)
    final = os.path.join(temp_dir, Path(slide_path).with_suffix(".tiff").name)
    tmp = final + ".tmp"
    img = pyvips.Image.new_from_file(slide_path, access="sequential")
    img.tiffsave(
        tmp,
        tile=True, pyramid=True,
        compression="jpeg", Q=JPEG_Q,
        bigtiff=True,
        tile_width=256, tile_height=256,
    )
    os.replace(tmp, final)
    return final, temp_dir


class WSIFeatureExtractor:
    """Drop-in replacement: .extract(wsi_path) -> [N, 1024] L2-normalized bag.

    NOTE: unlike the previous simplified version this needs the WSI *path*
    (openslide reads regions lazily), not a pre-loaded array. interf1/model.py
    passes the path.
    """

    def __init__(self, resnet_weights_path: str, device: str = "cpu",
                 batch_size: int = 128, num_workers: int = 4):
        import timm
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers if device != "cpu" else 0

        # timm resnet50.tv_in1k, features_only stage-3 (1024-d) + AdaptiveAvgPool,
        # built offline then loaded with the bundled torchvision ImageNet weights.
        model = timm.create_model(
            "resnet50.tv_in1k",
            features_only=True, out_indices=(3,),
            pretrained=False, num_classes=0,
        )
        state = torch.load(resnet_weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, _unexpected = model.load_state_dict(state, strict=False)
        # Expect 0 missing (layer4/fc are the only unexpected, and are unused).
        assert not missing, f"timm encoder missing weights: {missing[:5]}"
        self.model = model.to(device).eval()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # Same eval transform CLAM uses for resnet50_trunc.
        self.img_transforms = get_eval_transforms(
            mean=IMAGENET_MEAN, std=IMAGENET_STD, target_img_size=224,
        )

    def _encode(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(batch)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return self.pool(out).squeeze(-1).squeeze(-1)

    @torch.no_grad()
    def extract(self, wsi_path: str) -> torch.Tensor:
        work_path, cleanup = _ensure_pyramidal(str(wsi_path), tempfile.gettempdir())
        h5_dir = tempfile.mkdtemp(prefix="patch_")
        try:
            wsi_obj = WholeSlideImage(work_path)

            # seg_level: CLAM default = best level for 64x downsample.
            if len(wsi_obj.level_dim) == 1:
                seg_level = 0
            else:
                seg_level = wsi_obj.getOpenSlide().get_best_level_for_downsample(64)

            wsi_obj.segmentTissue(
                seg_level=seg_level,
                filter_params=FILTER_PARAMS,
                keep_ids=[], exclude_ids=[],
                **SEG_PARAMS,
            )

            # CLAM's process_contours treats save_path as a DIRECTORY and writes
            # <slide_name>.h5 inside it (name = stem of work_path).
            wsi_obj.process_contours(
                save_path=h5_dir,
                patch_level=PATCH_LEVEL,
                patch_size=PATCH_SIZE,
                step_size=STEP_SIZE,
                **PATCH_PARAMS,
            )
            h5_path = os.path.join(h5_dir, f"{wsi_obj.name}.h5")

            bag = self._extract_features(work_path, h5_path)
        finally:
            shutil.rmtree(h5_dir, ignore_errors=True)
            if cleanup:
                shutil.rmtree(cleanup, ignore_errors=True)

        # Match training: load_wsi_features(normalize=True) L2-normalizes rows.
        return F.normalize(bag.float(), p=2, dim=-1)

    @torch.no_grad()
    def _extract_features(self, slide_path: str, h5_path: str) -> torch.Tensor:
        import openslide
        from torch.utils.data import DataLoader

        wsi = openslide.open_slide(slide_path)
        try:
            dataset = Whole_Slide_Bag_FP(
                file_path=h5_path, wsi=wsi, img_transforms=self.img_transforms,
            )
            loader_kwargs = (
                {"num_workers": self.num_workers, "pin_memory": True}
                if self.device != "cpu" else {}
            )
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, **loader_kwargs)
            feats = []
            for data in loader:
                batch = data["img"].to(self.device, non_blocking=True)
                feats.append(self._encode(batch).cpu())
        finally:
            if hasattr(wsi, "close"):
                wsi.close()
        return torch.cat(feats, dim=0) if feats else torch.zeros((0, 1024))
