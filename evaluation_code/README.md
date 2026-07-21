# REG2026 Evaluation Method

Local Docker evaluation for REG2026 **metric A** (workflow reasoning) and **metric B** (visual grounding).

> **Icons in this README:** 📌 reference · ⚠️ warning · 🚫 do not · ✅ do

---

## Bundled sample data

> ⚠️ **The cases and predictions shipped inside [`data/`](data/) are taken from the REG2026 **training set** and exist solely as working examples. They are **not** the hidden test set used for official scoring.

The sample lets you verify the pipeline end-to-end without setting anything up. Once everything runs, replace the bundled files with your own cases and predictions (see [Building your own validation set](#building-your-own-validation-set) below).

---

## Building your own validation set

Use this package to simulate the official evaluation on a subset of training data your algorithm has **not** seen — giving you a realistic signal of how it will perform on the leaderboard.

### What you need

| Interface | What to prepare | Where to put it |
|-----------|----------------|-----------------|
| **Metric B** (interf0 — visual grounding) | `rois_mapping.txt` with your chosen ROIs | `data/cases/interf0/rois_mapping.txt` |
| **Metric A** (interf1 — workflow reasoning) | `ground_truth_CoT.json` with ground-truth chain-of-thought | `data/cases/interf1/ground_truth_CoT.json` |
| **Metric B predictions** | One `{ "id", "answer" }` per ROI | `data/predictions/interf0/predictions.json` |
| **Metric A predictions** | One `{ "id", "chain-of-thought" }` per WSI | `data/predictions/interf1/predictions.json` |

You may evaluate **one metric at a time** — provide only the interf0 files, only the interf1 files, or both.

### Step-by-step

**1. Prepare your reference cases** (ground truth)

- **Metric B:** Edit `data/cases/interf0/rois_mapping.txt`. Each row defines one ROI. The required tab-separated columns are:

  | Column | Values | Role |
  |--------|--------|------|
  | `anonymous_id` | unique string, e.g. `000000` | case key |
  | `image` | filename, e.g. `roi_000000.jpg` | image reference |
  | `variant` | `original` or `perturbed` | B2 sensitivity |
  | `paired_anonymous_id` | another `anonymous_id` or `none` | B2 pairing |
  | `b3_paired_anonymous_id` | another `anonymous_id` or `none` | B3 pairing |
  | `label` | `tissue` or `background` | B1 / B2 / B3 |
  | `question` | the question string | asked of the model |

  Extra columns (e.g. `image_path`, `slide_id`) are allowed and ignored. See [`data/cases/README.md`](data/cases/README.md) for a full explanation of B1/B2/B3 pairing.

  Optionally place the actual ROI JPEG thumbnails in `data/cases/interf0/` for your own reference — the prep scripts do **not** read image files.

- **Metric A:** Edit `data/cases/interf1/ground_truth_CoT.json`. Each entry is one WSI case:

  ```json
  [
    {
      "id": "YOUR_SLIDE.tiff",
      "chain-of-thought": [
        { "question": "...", "answer": "...", "next_question": "..." },
        { "question": "...", "answer": "...", "next_question": "" }
      ]
    }
  ]
  ```

  The last step must have `"next_question": ""`. Optionally place the actual `.tiff` WSI files in `data/cases/interf1/` for reference — they are not read by the prep scripts.

**2. Add your algorithm's predictions**

- **Metric B** (`data/predictions/interf0/predictions.json`): one entry per ROI, `id` must match `anonymous_id` in your `rois_mapping.txt`:

  ```json
  [
    { "id": "000000", "answer": "No tissue is visible in this ROI." }
  ]
  ```

- **Metric A** (`data/predictions/interf1/predictions.json`): one entry per WSI, `id` must match the `id` in your `ground_truth_CoT.json` (file extension optional):

  ```json
  [
    {
      "id": "YOUR_SLIDE.tiff",
      "chain-of-thought": [
        { "question": "...", "answer": "...", "next_question": "..." },
        { "question": "...", "answer": "...", "next_question": "" }
      ]
    }
  ]
  ```

**3. Run the pipeline**

```bash
cd submission_evaluation_code
./prepare_test_ground_truth.sh   # step 1 — cases → ground_truth/
./prepare_test_input.sh          # step 2 — predictions → test/input/
./download_model_weights.sh      # once only (~16 GB, skip if already done)
./do_test_run.sh                 # step 3 — Docker evaluation → test/output/metrics.json
```

The output score in `test/output/metrics.json` uses exactly the same scoring logic as the official leaderboard.

---

## Quick try (bundled sample)

Run this from `submission_evaluation_code/` to verify the pipeline using the included training-set samples and placeholder predictions:

```bash
cd submission_evaluation_code
./prepare_test_ground_truth.sh
./prepare_test_input.sh
./download_model_weights.sh   # once (~16 GB)
./do_test_run.sh
```

---

## Table of Contents

1. [Docker](#1-docker)
2. [How the container runs](#2-how-the-container-runs)
3. [What gets evaluated](#3-what-gets-evaluated)
4. [Repository layout](#4-repository-layout)
5. [Judge model weights](#5-judge-model-weights)
6. [Preparing data](#6-preparing-data)
7. [Configuration](#7-configuration)
8. [Local testing](#8-local-testing)
9. [Fixed paths](#9-fixed-paths)

---

## 1. Docker

Evaluation runs in a **Docker container**: Python code, dependencies, and runtime packaged together.

| Term | Meaning |
|---|---|
| **Image** | Built from the [`Dockerfile`](Dockerfile) via [`do_build.sh`](do_build.sh). Includes spaCy, PubMedBERT embeddings, and evaluation code. |
| **Ground truth** | Reference annotations and judge LLM weights on the host, mounted read-only at run time. |

---

## 2. How the container runs

Unlike an algorithm container (one case per run), the **evaluation container processes every job listed in** `/input/predictions.json` **in a single run**.

The platform (or [`do_test_run.sh`](do_test_run.sh) locally) mounts:

- **Predictions** at `/input` (including `predictions.json` and per-job output folders)
- **Ground truth** at `/opt/ml/input/data/ground_truth/`
- **Writable** `/output` for results

[`evaluate.py`](evaluate.py) reads submissions, compares them to ground truth, aggregates scores, writes **`/output/metrics.json`**, and exits.

In practice:

- **No network** at runtime — install Python deps in the image; download judge weights on the host before running (section 5).
- **One evaluation run** = all jobs in `predictions.json`.
- **Write results only under `/output`.**

---

## 3. What gets evaluated

| Metric | Interface | Submission file | Ground truth |
|--------|-----------|-----------------|--------------|
| **A** | Workflow reasoning (interf1) | `chain-of-thought.json` | [`ground_truth/metric_A/`](ground_truth/metric_A/) |
| **B** | Visual grounding (interf0) | `visual-context-response.json` | [`ground_truth/metric_B/`](ground_truth/metric_B/) |

**Sample pack** (bundled `test/input/`):

| Metric | Cases |
|--------|-------|
| A | 2 |
| B | 18 |

**20 jobs** total in [`test/input/predictions.json`](test/input/predictions.json), keyed by `pk` and mapped via [`ground_truth/manifest.json`](ground_truth/manifest.json).

### Score breakdown

**Metric A sub-scores** (combined into `workflow_final_ranking_score`):

| Sub-score | Key | Weight |
|-----------|-----|--------|
| Binary path validity | `binary_path_validity` | 5 % |
| Edge F1 | `edge_f1` | 30 % |
| MESS (non-final steps) | `mess` | 25 % |
| Final report score | `final_report_score` | 40 % |

**Metric B sub-scores** (combined into `visual_final_score`):

| Sub-score | Key | Weight |
|-----------|-----|--------|
| Background rejection | `background_rejection` | 30 % |
| Input sensitivity | `input_sensitivity` | 30 % |
| Cross-region consistency | `cross_region_consistency` | 40 % |

**Overall score** = 70 % × `workflow_final_ranking_score` + 30 % × `visual_final_score`

All scores are written to `test/output/metrics.json` under `aggregates`.

---

## 4. Repository layout

```
submission_evaluation_code/
├── README.md
├── evaluation_config.env      # settings (section 7)
├── config.sh                  # loads evaluation_config.env
├── download_model_weights.sh  # fetch judge LLM (section 5)
├── Dockerfile
├── requirements.txt
├── evaluate.py                # container entrypoint
├── evaluate_metrics.py        # metric A/B scoring logic
├── helpers.py
├── do_build.sh
├── do_test_run.sh
├── data/                      # cases, predictions, prep scripts
│   ├── README.md              # data preparation guide (start here)
│   ├── data_prep_lib.py       # shared prep helpers
│   ├── prepare_test_ground_truth.py
│   ├── prepare_test_input.py
│   ├── cases/                 # reference data (see data/cases/README.md)
│   │   ├── interf0/rois_mapping.txt        # metric B ROI table
│   │   └── interf1/ground_truth_CoT.json   # metric A ground-truth workflows
│   └── predictions/           # algorithm outputs (see data/predictions/README.md)
│       ├── interf0/predictions.json
│       └── interf1/predictions.json
├── prepare_test_ground_truth.sh
├── prepare_test_input.sh
├── ground_truth/
│   ├── metric_A/              # built by prepare_test_ground_truth.sh
│   ├── metric_B/              # built by prepare_test_ground_truth.sh
│   ├── manifest.json          # built by prepare_test_ground_truth.sh
│   └── Qwen3-8B/              # judge weights (after ./download_model_weights.sh)
└── test/
    ├── input/                 # built by prepare_test_input.sh
    └── output/                # metrics.json written by do_test_run.sh
```

Customize evaluation by editing [`evaluation_config.env`](evaluation_config.env), [`data/`](data/), [`ground_truth/`](ground_truth/), or the Dockerfile — not by changing path constants in `evaluate.py` unless you change the mount layout (section 9).

---

## 5. Judge model weights

Metric A uses a local **Qwen3-8B** judge (`LocalQwenJudgeLLM` in `evaluate_metrics.py`). Weights live under **`ground_truth/Qwen3-8B/`** on the host (~16 GB) and are **not** checked into git.

| Location | Purpose |
|----------|---------|
| `ground_truth/Qwen3-8B/` (host) | Store downloaded weights |
| `/opt/ml/input/data/ground_truth/Qwen3-8B/` (container) | Runtime path (`JUDGE_MODEL_PATH`) |

**Download (required before first test run):**

```bash
./download_model_weights.sh
```

The script loads [`evaluation_config.env`](evaluation_config.env), creates a temporary Python environment, installs `huggingface_hub`, and downloads `JUDGE_MODEL_REPO_ID` into `ground_truth/<JUDGE_MODEL_REL>/`. Re-running is safe: Hugging Face skips files that are already up to date.

Set optional **`HF_TOKEN`** in `evaluation_config.env` for faster / authenticated downloads. Keep this file private if your token is set.

---

## 6. Preparing data

This section summarizes the [**Building your own validation set**](#building-your-own-validation-set) flow at the top of this README. For file formats, custom case sets, ID rules, and optional CLI paths, use **[`data/README.md`](data/README.md)**.

| Step | Script | Output |
|------|--------|--------|
| Ground truth | `./prepare_test_ground_truth.sh` | `ground_truth/metric_A/`, `metric_B/`, `manifest.json` |
| Test inputs | `./prepare_test_input.sh` | `test/input/` |

**Source paths (under `data/`):**

- `data/cases/interf0/rois_mapping.txt` — metric B reference ROIs
- `data/cases/interf1/ground_truth_CoT.json` — metric A reference workflows
- `data/predictions/interf0/predictions.json` — visual answers `[{"id":"<pk>","answer":"..."}]`
- `data/predictions/interf1/predictions.json` — workflow steps `[{"id":"<pk>","chain-of-thought":[...]}]`

You may provide **interf0 only**, **interf1 only**, or both. Deeper reference: [`data/cases/README.md`](data/cases/README.md), [`data/predictions/README.md`](data/predictions/README.md).

---

## 7. Configuration

All defaults are in **[`evaluation_config.env`](evaluation_config.env)**. [`config.sh`](config.sh) sources it; [`do_build.sh`](do_build.sh) and [`do_test_run.sh`](do_test_run.sh) pass the same values into the container.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DOCKER_IMAGE_TAG` | `reg26-evaluation-sample` | Local image name |
| `JUDGE_MODEL_REPO_ID` | `Qwen/Qwen3-8B` | Repo for `download_model_weights.sh` |
| `JUDGE_MODEL_REL` | `Qwen3-8B` | Subfolder under `ground_truth/` |
| `JUDGE_MODEL_PATH` | `/opt/ml/input/data/ground_truth/Qwen3-8B` | Judge path inside container |
| `JUDGE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `EMBEDDING_MODEL` | `NeuML/pubmedbert-base-embeddings` | Sentence embedding model |
| `HF_TOKEN` | *(set yours or leave empty)* | Optional Hugging Face token for faster downloads |
| `USE_GPUS` | `auto` | Local Docker GPU (`auto` / `1` / `0`) |
| `INCLUDE_PER_CASE_RESULTS` | `1` | Per-case rows in `metrics.json` |
| `INCLUDE_EVALUATION_DETAILS` | `0` | Verbose `details` block |

> ⚠️ **`HF_TOKEN`** — if you set a real Hugging Face token in `evaluation_config.env`, keep the file private and do not commit it.

---

## 8. Local testing

Run from `submission_evaluation_code/`:

```bash
./prepare_test_ground_truth.sh  # when cases change
./prepare_test_input.sh         # when predictions change
./download_model_weights.sh     # once (section 5)
./do_test_run.sh
```

[`do_test_run.sh`](do_test_run.sh):

1. Builds the image ([`do_build.sh`](do_build.sh)).
2. Runs the container with `--network none`, mounting `test/input/` and `ground_truth/`.
3. Writes [`test/output/metrics.json`](test/output/metrics.json).

**Prerequisites:** Docker, Python 3 (host, for download script), judge weights under `ground_truth/Qwen3-8B/`, prepared `test/input/` (see [Building your own validation set](#building-your-own-validation-set)).

**Verbose output:**

```bash
INCLUDE_PER_CASE_RESULTS=1 INCLUDE_EVALUATION_DETAILS=1 ./do_test_run.sh
```

---

## 9. Fixed paths

Used by [`evaluate.py`](evaluate.py). Do not change these unless you change the mount layout.

| Path | Purpose |
|------|---------|
| `/input/predictions.json` | Job list and output locations |
| `/input/<pk>/output/...` | Per-job algorithm outputs |
| `/opt/ml/input/data/ground_truth/` | Reference data + judge weights |
| `/opt/ml/input/data/ground_truth/manifest.json` | `pk` → case `title` |
| `/opt/ml/input/data/ground_truth/Qwen3-8B/` | Judge LLM |
| `/output/metrics.json` | Evaluation results |
