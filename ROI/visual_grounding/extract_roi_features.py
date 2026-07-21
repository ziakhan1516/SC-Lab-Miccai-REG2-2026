"""Stage 1 of Visual Grounding (Metric B): turn each ROI .jpg into a CONCH
feature so the WSI reasoning model can be asked about it.

The reasoning model consumes a bag of patch features shaped [N_patches, 512]
(CONCH ViT-B-16). An ROI is a single 256x256 patch, so each ROI becomes a
[1, 512] tensor saved as <out_dir>/<roi_id>.pt  -- exactly the same on-disk
format as the WSI .pt files, so the downstream loader treats it identically.

Self-contained: vendors the `conch` package under _vendor/ and loads the
local CONCH weights, so it does not depend on any conda env or the main
training code.

Run (project's python3 already has torch+timm+PIL):
    python3 extract_roi_features.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROI_DIR = HERE.parent                       # .../Try2/ROI
DEFAULT_WEIGHTS = HERE.parent.parent / "ClamforMiccai2026" / "pytorch_model.bin"

# Make the vendored conch package importable without touching any conda env.
sys.path.insert(0, str(HERE / "_vendor"))


def load_conch(weights_path: str, device: str):
    """Load CONCH ViT-B-16 and its eval preprocessing transform."""
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", weights_path)
    model.eval().to(device)
    return model, preprocess


@torch.no_grad()
def encode_image(model, tensor: torch.Tensor) -> torch.Tensor:
    """Return the 512-d CONCH patch embedding (pre-projection, un-normalised),
    matching how CLAM extracts WSI patch features (proj_contrast=False,
    normalize=False). L2-normalisation is applied later at load time, exactly
    like the WSI .pt files."""
    return model.encode_image(tensor, proj_contrast=False, normalize=False)


def main():
    ap = argparse.ArgumentParser(description="Extract CONCH features for ROI patches.")
    ap.add_argument("--rois-dir", default=str(ROI_DIR / "rois"),
                    help="Folder of roi_*.jpg images.")
    ap.add_argument("--pairs-json", default=str(ROI_DIR / "roi_question_pairs.json"),
                    help="roi_question_pairs.json (drives which ROIs to encode).")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="CONCH pytorch_model.bin.")
    ap.add_argument("--out-dir", default=str(HERE / "roi_features"),
                    help="Where per-ROI [1,512] .pt files are written.")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if omitted).")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pairs_json) as f:
        pairs = json.load(f)

    print(f"CONCH weights : {args.weights}")
    print(f"ROIs          : {len(pairs)} from {args.rois_dir}")
    print(f"Output        : {out_dir}")
    print(f"Device        : {device}\nLoading CONCH ...")

    from PIL import Image
    model, preprocess = load_conch(args.weights, device)

    manifest = []
    for i, item in enumerate(pairs, 1):
        roi_id = item["id"]
        img_path = Path(args.rois_dir) / item["image"]
        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        feat = encode_image(model, tensor).squeeze(0).cpu().float()   # [512]
        feat = feat.unsqueeze(0)                                      # [1, 512]

        out_path = out_dir / f"{roi_id}.pt"
        torch.save(feat, out_path)
        manifest.append({
            "id": roi_id,
            "image": item["image"],
            "question": item.get("question", ""),
            "feature_path": str(out_path),
            "shape": list(feat.shape),
        })
        print(f"[{i}/{len(pairs)}] {item['image']} -> {out_path.name}  {tuple(feat.shape)}")

    manifest_path = out_dir / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest)} ROI features written. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
