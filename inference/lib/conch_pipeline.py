"""CONCH feature extraction for the REG2026 container (both interfaces).

The generation model (DeepSeek-R1-Distill-Qwen-1.5B + Perceiver resampler) was
trained on **CONCH ViT-B-16** patch features (512-d), so here CONCH is the *only*
visual encoder — used identically for:

  - Interface 0 (ROI): one .jpeg patch  -> [1, 512]
  - Interface 1 (WSI): pyramid/seg/patch -> [N, 512]

The WSI seg/patch/pyramid logic is reused from `feature_extractor.py` (which
already mirrors the training-time `wsi_full_pipeline.py`, including the
non-pyramidal -> pyvips tiffsave conversion). Only the encoder is swapped from
resnet50 to CONCH. Features are L2-normalised per patch, matching training
(`load_wsi_features(normalize=True)`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

# NOTE: the WSI machinery (feature_extractor / wsi_core / dataset_modules) is
# imported lazily inside extract_wsi(), so the ROI path (Interface 0) does not
# require openslide / pyvips to be importable.


def _load_conch(conch_ckpt_path: str, device: str, target_img_size: int = 224):
    """Load CONCH ViT-B-16 via the vendored CLAM builder.

    get_encoder('conch_v1') sets model.forward = encode_image(proj_contrast=False,
    normalize=False) and returns the OpenAI-normalised eval transform, so
    model(batch) -> [B, 512] exactly like the training feature extractor.
    """
    os.environ["CONCH_CKPT_PATH"] = str(conch_ckpt_path)
    from models import get_encoder  # vendored lib/clam/models
    model, img_transforms = get_encoder("conch_v1", target_img_size=target_img_size)
    model = model.eval().to(device)
    return model, img_transforms


class CONCHEncoder:
    """Shared CONCH encoder: ROI images and WSI patch bags -> 512-d features."""

    def __init__(self, conch_ckpt_path: str, device: str = "cpu",
                 batch_size: int = 256, num_workers: int = 8,
                 max_patches: int | None = None):
        self.device = device
        # Larger batch + more workers => higher patch throughput on GPU. These
        # are pure I/O / throughput knobs and do NOT change the extracted
        # features. Env-overridable so the platform can tune to its hardware.
        self.batch_size = int(os.environ.get("REG_CONCH_BATCH", batch_size))
        env_workers = int(os.environ.get("REG_NUM_WORKERS", num_workers))
        self.num_workers = env_workers if device != "cpu" else 0
        # Upper bound on patches per slide. The Perceiver resampler compresses the
        # whole bag into a fixed number of latent tokens, so for an outlier slide
        # with tens of thousands of patches a uniform (evenly strided) subset
        # preserves tissue coverage while bounding extraction time -- this is the
        # single-case timeout fix. Set generously so ordinary slides are never
        # touched; env-tunable (REG_MAX_PATCHES=0 disables the cap).
        if max_patches is None:
            max_patches = int(os.environ.get("REG_MAX_PATCHES", 12000))
        self.max_patches = max_patches if max_patches and max_patches > 0 else None
        self.model, self.img_transforms = _load_conch(conch_ckpt_path, device)

    # ---------------------------------------------------------------- ROI ---
    @torch.no_grad()
    def encode_roi(self, roi_image) -> torch.Tensor:
        """PIL RGB ROI -> [1, 512] L2-normalised (one-patch bag)."""
        x = self.img_transforms(roi_image).unsqueeze(0).to(self.device)
        feat = self.model(x).float().cpu()            # [1, 512]
        return F.normalize(feat, p=2, dim=-1)

    # ---------------------------------------------------------------- WSI ---
    @torch.no_grad()
    def extract_wsi(self, wsi_path: str) -> torch.Tensor:
        """WSI .tiff -> [N, 512] L2-normalised bag. Handles non-pyramidal slides."""
        # Reuse the exact WSI pyramid/seg/patch machinery used for resnet
        # features (imported here so Interface 0 never needs openslide/pyvips).
        from feature_extractor import (
            _ensure_pyramidal, SEG_PARAMS, FILTER_PARAMS, PATCH_PARAMS,
            PATCH_SIZE, STEP_SIZE, PATCH_LEVEL,
        )
        from wsi_core.WholeSlideImage import WholeSlideImage

        work_path, cleanup = _ensure_pyramidal(str(wsi_path), tempfile.gettempdir())
        h5_dir = tempfile.mkdtemp(prefix="patch_")
        try:
            wsi_obj = WholeSlideImage(work_path)
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
            wsi_obj.process_contours(
                save_path=h5_dir,
                patch_level=PATCH_LEVEL,
                patch_size=PATCH_SIZE,
                step_size=STEP_SIZE,
                **PATCH_PARAMS,
            )
            h5_path = os.path.join(h5_dir, f"{wsi_obj.name}.h5")
            bag = self._features_from_h5(work_path, h5_path)
        finally:
            shutil.rmtree(h5_dir, ignore_errors=True)
            if cleanup:
                shutil.rmtree(cleanup, ignore_errors=True)

        return F.normalize(bag.float(), p=2, dim=-1)

    @torch.no_grad()
    def _features_from_h5(self, slide_path: str, h5_path: str) -> torch.Tensor:
        import openslide
        from torch.utils.data import DataLoader
        from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP

        wsi = openslide.open_slide(slide_path)
        try:
            dataset = Whole_Slide_Bag_FP(
                file_path=h5_path, wsi=wsi, img_transforms=self.img_transforms,
            )
            # Cap an outlier slide to at most max_patches via a deterministic,
            # evenly strided subset of coordinates. Done BEFORE the DataLoader so
            # the dropped patches are never read or encoded (that read+encode is
            # where the per-case time goes). Normal slides (<= cap) are untouched.
            if self.max_patches is not None and len(dataset) > self.max_patches:
                from torch.utils.data import Subset
                n = len(dataset)
                stride = n / float(self.max_patches)
                idx = [int(i * stride) for i in range(self.max_patches)]
                # Guard against rounding collisions / out-of-range indices.
                idx = sorted({min(j, n - 1) for j in idx})
                print(f"[WSI] {n} patches > cap {self.max_patches}; "
                      f"uniformly subsampling to {len(idx)} patches.")
                dataset = Subset(dataset, idx)
            loader_kwargs = (
                {"num_workers": self.num_workers, "pin_memory": True}
                if self.device != "cpu" else {}
            )
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, **loader_kwargs)
            feats = []
            for data in loader:
                batch = data["img"].to(self.device, non_blocking=True)
                feats.append(self.model(batch).float().cpu())
        finally:
            if hasattr(wsi, "close"):
                wsi.close()
        return torch.cat(feats, dim=0) if feats else torch.zeros((0, 512))
