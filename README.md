# SC-Lab — REG2026 Grand Challenge Submission

Code, training pipeline, and Docker inference container for our submission to the
**REG2026** (Reasoning-Enhanced Generation for pathology) Grand Challenge.

---

## Team

| | |
|---|---|
| **Team name** | SC-Lab |
| **Team members** | Ziaullah Khan · Shrestha Nischal Lal · Nheng Vanchhay · Md. Ariful Islam Mozumder · Hee-Cheol Kim |
| **Grand Challenge username** | `zkhan.msee19seecs` (SC-LAB) |
| **Grand Challenge profile URL** | https://grand-challenge.org/users/zkhan.msee19seecs/ |

---

## Method (brief)

We tackle both challenge interfaces with a single **vision-conditioned reasoning
LLM**:

- **Vision encoder — CONCH (ViT-B/16), feature extraction only.** Region-of-interest
  thumbnails (Interface 0) and whole-slide images (Interface 1) are encoded into
  512-dim patch features. WSIs are first segmented and patched with the CLAM
  pipeline, then each patch is encoded by CONCH.
- **Visual resampler — Perceiver.** The variable-length bag of CONCH features is
  compressed into **128 visual tokens** (depth 3, 8 heads) that condition the LLM.
- **Reasoning LLM — DeepSeek-R1-Distill-Qwen-1.5B + LoRA.** LoRA adapters
  (r=16, α=32, dropout=0.05, on all attention + MLP projections) are trained on
  chain-of-thought pathology reasoning data. bf16.
- **Interface 0 (Visual Grounding):** ROI → CONCH `[1, 512]` → LLM → answer string.
- **Interface 1 (Workflow Reasoning):** WSI → CLAM seg/patch → CONCH `[N, 512]`
  → LLM → chain-of-thought steps (parsed into the required `{question, answer,
  next_question}` array).

**Training:** 3 epochs, lr 2e-5, effective batch 32 (batch 1 × grad-accum 16 ×
2 GPUs), DDP + gradient checkpointing. Data: `train_CoT.json` → 10,925 train /
223 held-out test (2% split, seed 42), matched to CONCH features.

**Held-out test results (223 cases):** Workflow Reasoning Score **0.721**
(BPV 0.565, Edge-F1 0.807, MESS 0.756, Final Report 0.655).

Full write-up: [`REG2026_paper.pdf`](REG2026_paper.pdf) · architecture figure:
[`model_overview.pdf`](model_overview.pdf).

---

## Repository layout

```
.
├── README.md                    # this file
├── requirements-train.txt       # training / evaluation environment
├── REG2026_paper.pdf / .tex     # method write-up
│
├── Training code (repo root)
│   ├── train.py                 # DDP training entry point
│   ├── multimodal_alignment.py  # WSIReportGenerator (CONCH + Perceiver + LoRA LLM)
│   ├── multimodal_dataset.py    # dataset + collate_fn
│   ├── reasoning_mllm.py        # reasoning / generation model
│   ├── seq2seq_alignment.py     # sequence alignment components
│   ├── attention_mil.py         # attention-MIL aggregator
│   ├── build_wsi_embeddings.py  # CONCH feature extraction for WSIs
│   ├── text_encoder.py / text_preprocess.py / json_loader.py
│   └── main.py                  # orchestration / config
│
├── Evaluation code (repo root)
│   ├── evaluate_test_set.py         # held-out test evaluation
│   ├── evaluate_report_metrics.py   # report-level metrics
│   ├── evaluate_workflow_reasoning.py
│   ├── generate_and_compare.py / metrics.py / manual_check.py
│
├── inference/               # Docker container used for challenge submission
│   ├── Dockerfile               # CUDA 12.8, non-root, offline
│   ├── inference.py             # ENTRYPOINT — dispatches by interface
│   ├── core.py                  # fixed platform I/O (do not edit)
│   ├── src/interf0/model.py     # Interface 0 handler
│   ├── src/interf1/model.py     # Interface 1 handler
│   ├── lib/                     # vendored CONCH, CLAM, generation model
│   ├── prepare_model_dir.py     # builds the mounted model/ dir from checkpoints
│   ├── requirements.txt
│   └── do_build.sh / do_test_run.sh / do_save.sh
│
├── evaluation_code/         # official-style scoring harness
│   ├── evaluate.py / evaluate_metrics.py / helpers.py
│   └── requirements.txt
│
├── ROI/                     # Visual-grounding (Interface 0) tooling
└── checkpoints/             # config + metrics JSON (weights hosted externally)
```

> **Not in git:** trained weights, Docker image tarballs, the LLM judge model,
> and slide images are excluded (see `.gitignore`). Download instructions below.

---

## Environment setup

Two environments are used (a training/eval env and the container env).

### Training / evaluation env

```bash
conda create -n reg2026 python=3.10 -y
conda activate reg2026

# PyTorch (CUDA 12.8) — must match the trained LoRA adapter
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements-train.txt
```

> The pinned `transformers==4.57.6` / `peft==0.17.1` versions are **required** —
> older `peft` cannot load the saved LoRA adapters.

### Inference container env

The submission runs inside Docker; all dependencies are pinned in
[`inference/Dockerfile`](inference/Dockerfile) and
[`inference/requirements.txt`](inference/requirements.txt). No manual setup needed
beyond Docker + the NVIDIA container toolkit.

---

## Model weights

The trained weights are too large for GitHub and are hosted externally.

**Download:** _link to be added — upload the `model/` directory (Google Drive /
OneDrive / Dropbox / HuggingFace) and paste the URL here._

Contents (≈4.4 GB after `prepare_model_dir.py`):

| Component | What it is |
|---|---|
| `base_llm/`   | DeepSeek-R1-Distill-Qwen-1.5B base (bf16, bundled for offline use) |
| `generation/` | LoRA adapter + Perceiver resampler (`resampler.pt`) |
| `conch/`      | CONCH ViT-B/16 encoder (`pytorch_model.bin`) |

Place the extracted `model/` so it is mounted at `/opt/ml/model` at container
runtime (see `inference/prepare_model_dir.py` and `do_test_run.sh`).

---

## Reproducing inference (submission container)

```bash
cd inference

# 1. Build the model/ directory from the downloaded checkpoints
python prepare_model_dir.py

# 2. Build the Docker image
./do_build.sh

# 3. Run the container on the sample test input (test/input, both interfaces)
./do_test_run.sh
#   -> writes test/output/interf0/visual-context-response.json
#   -> writes test/output/interf1/chain-of-thought.json

# 4. Export the image tarball for Grand Challenge upload
./do_save.sh
```

A tiny sample Interface-0 input ships in `inference/test/input/`. Interface-1
requires a whole-slide `.tiff` (obtain from the challenge data — slides are not
redistributed here).

---

## Reproducing training

```bash
conda activate reg2026

# CONCH features for the WSIs (produces the feature bank used by training)
python build_wsi_embeddings.py

# Train the vision-conditioned reasoning LLM (DDP, 2 GPUs)
torchrun --nproc_per_node=2 train.py

# Evaluate on the held-out test split
python evaluate_test_set.py
```

Training config is captured in
`checkpoints/wsi_reasoning_r1qwen1p5b_conch/training_args.json` and
`model_config.json`; test metrics in `test_eval_v3.json`.

---

## License / data

Slide images and challenge data are governed by the REG2026 Grand Challenge terms
and are **not** redistributed in this repository.
