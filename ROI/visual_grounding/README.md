# Visual Grounding (Metric B) — ROI pipeline

Self-contained pipeline that asks the trained WSI reasoning model ROI-level
questions and records its answers, for the Visual Grounding metric (B1
background rejection, B2 input sensitivity, B3 cross-region consistency).

It does **not** touch the main training code — it only *imports* the existing
model loader (`load_generator`, `load_wsi_features`) from the parent `Try2`
folder and reuses the ROI data in `Try2/ROI/`.

## Data (in `Try2/ROI/`)
- `rois/roi_*.jpg` — 18 ROI patches (256x256, 5x).
- `roi_question_pairs.json` — `{id, image, question}` per ROI.
- `anonymous_rois_mapping.txt` — metadata: `label` (tissue|background),
  `variant` (original|perturbed), and the B2/B3 pairings.

## Pipeline

### Stage 1 — CONCH features (`extract_roi_features.py`)
Each ROI `.jpg` → a `[1, 512]` CONCH ViT-B-16 feature (one patch), saved as
`roi_features/<id>.pt`. Same on-disk format as the WSI `.pt` files, so the model
treats an ROI as a 1-patch slide. CONCH weights:
`Try2/ClamforMiccai2026/pytorch_model.bin`. The `conch` package is vendored
under `_vendor/` so no conda env is required.

```bash
# default python3 has torch+timm+PIL; conch is vendored
cd Try2/ROI/visual_grounding
python3 extract_roi_features.py
```
Output: `roi_features/*.pt` + `roi_features/feature_manifest.json`. (Already run.)

### Stage 2 — ask the model (`run_roi_grounding.py`)
Loads the trained **CONCH** reasoning model (`feature_dim=512`), asks each ROI
question, and writes `roi_grounding_answers.json` with the answer + ROI metadata
(label/variant/pairings) needed for B1/B2/B3.

```bash
# run in the SAME env used to train the model (loads the LoRA adapter).
# e.g. peft>=0.17:  /home/ali/storage2/anaconda3/envs/manga/bin/python
python3 run_roi_grounding.py \
  --checkpoint /home/ali/storage1/Bin-Version2/Reg2/codings/Try2/checkpoints/wsi_reasoning_r1qwen1p5b_conch \
  --roi-system-prompt
```
`--roi-system-prompt` swaps the heavy WSI report system prompt for a brief,
ROI-grounded one (recommended for these short questions).

> The CONCH model must be trained first (arch=reasoning, feature-dim 512 — see
> `Try2/helpingCommands.txt` block 1b). The empty `..._conch` checkpoint only has
> `splits/` so far.

## Environments (important)
- **Stage 1** runs in the default `python3` (torch 2.1.2, timm, PIL; `conch`
  vendored here).
- **Stage 2** must run in the env whose `peft` can load the saved LoRA adapter
  (the checkpoints were saved with `peft>=0.17`, e.g. the `manga` env). The
  default `python3` (peft 0.13.2) fails on `alora_invocation_tokens`.

## Next step (not yet implemented)
Scoring B1/B2/B3 over `roi_grounding_answers.json` needs a local pathology
**judge** model (semantic-equivalence / claim detection). The answers file
already carries `label`, `variant`, `pair_id`, and the paired IDs so the judge
can be plugged in directly.
```
B1 = mean correct-rejection over background ROIs
B2 = mean semantic-equivalence(original, perturbed) over tissue pairs
B3 = mean (1 - semantic-equivalence(tissue, background)) over cross pairs
Visual Grounding = 0.30*B1 + 0.30*B2 + 0.40*B3
```
