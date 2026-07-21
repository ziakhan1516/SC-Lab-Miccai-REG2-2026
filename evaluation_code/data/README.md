# Evaluation data preparation

Self-contained staging and conversion for local evaluation.

> ⚠️ **Bundled files are training-set samples.** The files shipped under [`cases/`](cases/) and [`predictions/`](predictions/) are taken from the REG2026 training set and exist only to let you verify the pipeline works out of the box. Replace them with your own cases and predictions when you want to measure your model's actual performance.

Edit **`cases/`** (reference data) and **`predictions/`** (algorithm outputs), then run the prep scripts to produce the parent package's **`ground_truth/`** and **`test/input/`** (consumed by `evaluate.py` and `./do_test_run.sh`).

**Run from package root** (`submission_evaluation_code/`):

```bash
cd submission_evaluation_code
./prepare_test_ground_truth.sh        # step 1 — cases → ground_truth/
./prepare_test_input.sh               # step 2 — predictions → test/input/
./do_test_run.sh                      # step 3 — Docker evaluation
```

The root wrapper scripts call the Python prep code under `data/`; you do not need to `cd data` first.

---

## Building your own validation set

To evaluate your algorithm on a custom subset of training data:

1. **Define reference cases** (ground truth) — edit or replace the source files in `cases/`.
2. **Run step 1** so `ground_truth/manifest.json` is regenerated with your new cases.
3. **Add predictions** whose `id` fields match those cases.
4. **Run step 2** to build `test/input/`; only cases that have a matching prediction entry become evaluation jobs.

| Interface | Reference (you define) | Predictions (your algorithm) | Metric |
|-----------|----------------------|------------------------------|--------|
| **interf0** | [`cases/interf0/rois_mapping.txt`](cases/interf0/rois_mapping.txt) | [`predictions/interf0/predictions.json`](predictions/interf0/predictions.json) | B — visual grounding |
| **interf1** | [`cases/interf1/ground_truth_CoT.json`](cases/interf1/ground_truth_CoT.json) | [`predictions/interf1/predictions.json`](predictions/interf1/predictions.json) | A — workflow reasoning |

You do **not** need to hand-edit `ground_truth/manifest.json` or `test/input/predictions.json` — the scripts generate them automatically.

Provide **at least one** interface in step 1 and **at least one** prediction file in step 2. Omit an interface entirely by removing (or not creating) its case file and prediction file.

More field-level detail: [`cases/README.md`](cases/README.md), [`predictions/README.md`](predictions/README.md).

---

## Linking cases, predictions, and manifest

Everything is keyed by **`pk`** — a case identifier with no file extension. Prep scripts normalize `id` values the same way (`PIT_01_00020_01.tiff` → `PIT_01_00020_01`).

```
cases/interf0/rois_mapping.txt          cases/interf1/ground_truth_CoT.json
         │                                        │
         │  anonymous_id / id                     │
         ▼                                        ▼
              prepare_test_ground_truth.sh
                         │
                         ▼
              ground_truth/manifest.json  ←── one entry per case (auto-generated)
              ground_truth/metric_B/...       ground_truth/metric_A/...
                         │
predictions/interf0|interf1/predictions.json
         │  "id" must match manifest pk
         │  (interf0: id must also be in rois_mapping)
         ▼
              prepare_test_input.sh
                         │
                         ▼
              test/input/predictions.json  +  test/input/<pk>/output/*.json
```

| Check | Rule |
|-------|------|
| Step 1 before step 2 | `prepare_test_input.sh` fails without `ground_truth/manifest.json`. |
| Metric B prediction `id` | Must equal `anonymous_id` from `rois_mapping.txt` (after normalization) **and** appear in `manifest.json`. |
| Metric A prediction `id` | Must match an `id` from `ground_truth_CoT.json` (extension optional) **and** appear in `manifest.json`. |
| Jobs evaluated | Only cases listed in `data/predictions/.../predictions.json` are written to `test/input/predictions.json`. Reference cases without a prediction are not run locally (but remain in ground truth). |
| Duplicate ids | Duplicate `id` in one prediction file → `DataPrepError`. |

---

## Directory layout

```
submission_evaluation_code/
├── prepare_test_ground_truth.sh       # wrapper script at package root
├── prepare_test_input.sh              # wrapper script at package root
└── data/
    ├── README.md                      # this file
    ├── data_prep_lib.py               # shared helpers
    ├── prepare_test_ground_truth.py   # step 1 implementation
    ├── prepare_test_input.py          # step 2 implementation
    ├── cases/                         # see cases/README.md
    │   ├── interf0/rois_mapping.txt
    │   └── interf1/ground_truth_CoT.json
    └── predictions/                   # see predictions/README.md
        ├── interf0/predictions.json
        └── interf1/predictions.json
```

---

## Optional interfaces

You may provide **interf0 only**, **interf1 only**, or **both**:

| | interf0 only | interf1 only | both |
|---|-------------|-------------|------|
| `cases/` | `interf0/rois_mapping.txt` | `interf1/ground_truth_CoT.json` | both |
| `predictions/` | `interf0/predictions.json` | `interf1/predictions.json` | both |

- Missing case files → that interface is skipped when building `ground_truth/`.
- Missing prediction files → treated as zero jobs for that interface.
- At least **one** interface must be present in each step.

Errors raise **`DataPrepError`** with a short message (invalid JSON, unknown id, duplicate pk, etc.).

---

## Step 1 — Test ground truth (`prepare_test_ground_truth.sh`)

**Script:** [`prepare_test_ground_truth.py`](prepare_test_ground_truth.py) — run from package root: `./prepare_test_ground_truth.sh`

**Reads** (each optional; at least one required)

| File | Purpose |
|------|---------|
| [`cases/interf0/rois_mapping.txt`](cases/interf0/rois_mapping.txt) | Metric B ROI table |
| [`cases/interf1/ground_truth_CoT.json`](cases/interf1/ground_truth_CoT.json) | Metric A reference workflows |

**Writes** (under `../ground_truth/`)

| File | If interface omitted |
|------|----------------------|
| `metric_B/rois_mapping.txt` | Header-only TSV (no rows) |
| `metric_A/chain-of-thoughts-ground-truth.json` | `[]` |
| `manifest.json` | Entries for provided interfaces only (auto-generated — do not hand-edit) |

### Custom case files (optional CLI)

Default paths are under `data/cases/`. To point at files elsewhere:

```bash
./prepare_test_ground_truth.sh \
  --interf0-mapping /path/to/my_rois_mapping.txt \
  --interf1-cot /path/to/my_ground_truth_CoT.json
```

---

## Step 2 — Local test inputs (`prepare_test_input.sh`)

**Script:** [`prepare_test_input.py`](prepare_test_input.py) — run from package root: `./prepare_test_input.sh`

Run **after** step 1 so `../ground_truth/manifest.json` exists.

**Reads** (each optional; at least one required)

| File | Format |
|------|--------|
| [`predictions/interf0/predictions.json`](predictions/interf0/predictions.json) | `[{ "id": "<pk>", "answer": "..." }]` |
| [`predictions/interf1/predictions.json`](predictions/interf1/predictions.json) | `[{ "id": "<pk>", "chain-of-thought": [...] }]` |

- Each `id` must match a `display_set.pk` in `manifest.json` (extensions stripped).
- interf0 `id` must exist in `metric_B/rois_mapping.txt` when interf0 predictions are provided.

**Writes** (under `../test/input/`)

| Path | Content |
|------|---------|
| `predictions.json` | Job list (pk, input/output socket slugs + output paths) |
| `<pk>/output/visual-context-response.json` | JSON string answer (interf0) |
| `<pk>/output/chain-of-thought.json` | Step array (interf1) |

### Custom prediction files (optional CLI)

```bash
./prepare_test_input.sh \
  --interf0 /path/to/my_interf0_predictions.json \
  --interf1 /path/to/my_interf1_predictions.json \
  --manifest ../ground_truth/manifest.json
```

---

## ID / pk conventions

| Interface | Case key in `cases/` | `id` in predictions | `manifest` `pk` | `test/input/` folder |
|-----------|------------------------|---------------------|-----------------|----------------------|
| interf0 | `anonymous_id` in `rois_mapping.txt` | same (e.g. `000000`) | same | `test/input/000000/` |
| interf1 | `id` in `ground_truth_CoT.json` | same, extension allowed (e.g. `PIT_01_00020_01.tiff`) | extension stripped | `test/input/PIT_01_00020_01/` |

`id` values in prediction JSON may include extensions; scripts normalize them to extension-free `pk` for folders and `predictions.json`.

---

## Example prediction snippets

**interf0** (`predictions/interf0/predictions.json`):

```json
[
  { "id": "000000", "answer": "Yes, tissue is visible in this ROI." }
]
```

**interf1** (`predictions/interf1/predictions.json`):

```json
[
  {
    "id": "PIT_01_00020_01.tiff",
    "chain-of-thought": [
      {
        "question": "What is the organ?",
        "answer": "Breast",
        "next_question": "Is there any abnormality present?"
      },
      {
        "question": "What is the final pathology report?",
        "answer": "Breast, core needle biopsy; ...",
        "next_question": ""
      }
    ]
  }
]
```

The last workflow step must have `"next_question": ""`.

---

## Common errors

| Message | Typical fix |
|---------|-------------|
| `not in manifest.json` | Run step 1 after editing cases, or fix prediction `id` to match a case you defined. |
| `not in rois_mapping.txt` (interf0) | Add the ROI row in `cases/interf0/rois_mapping.txt` or remove the stray prediction. |
| `Missing manifest.json` | Run `./prepare_test_ground_truth.sh` before `./prepare_test_input.sh`. |
| `No case data found` | Add at least one of the two case source files. |
| `No predictions to export` | Add at least one prediction file with a non-empty array (or create the file). |
| `Mapping txt missing columns` | Your `rois_mapping.txt` is missing one or more required columns — see [`cases/README.md`](cases/README.md). |
