# Case data (ground truth sources)

> ⚠️ **Bundled files are training-set samples.** The `rois_mapping.txt` and `ground_truth_CoT.json` shipped here are taken from the REG2026 training set. Replace them with your own cases when you want to evaluate your algorithm on data it has not seen.

Place **at least one** interface below (or replace the sample files entirely with your own case set), then run [`../../prepare_test_ground_truth.sh`](../../prepare_test_ground_truth.sh) from the package root.

Overview and how prediction `id` values link to these files: [`../README.md`](../README.md#linking-cases-predictions-and-manifest). Package-level summary and step-by-step: [`../../README.md`](../../README.md#building-your-own-validation-set).

---

## interf0 — Metric B (visual grounding)

**Required file:** `interf0/rois_mapping.txt`

A **tab-separated** file. The first line must be a header. The following columns are **required** (extra columns are ignored):

| Column | Expected values | Role in evaluation |
|--------|-----------------|--------------------|
| `anonymous_id` | unique string, e.g. `000000` | case key; must match prediction `id` |
| `image` | filename, e.g. `roi_000000.jpg` | image reference |
| `variant` | `original` or `perturbed` | drives B2 input-sensitivity metric |
| `paired_anonymous_id` | another `anonymous_id` or `none` | B2: links original ↔ perturbed pair |
| `b3_paired_anonymous_id` | another `anonymous_id` or `none` | B3: links tissue ROI → background ROI |
| `label` | `tissue` or `background` | drives B1, B2, B3 routing |
| `question` | question text | passed to the model and judge |

### ROI images

Optionally place the actual ROI JPEG thumbnails (e.g. `roi_000000.jpg`) in `interf0/`. The prep scripts do **not** read image files — they read only `rois_mapping.txt`.

If interf0 is omitted entirely, metric B ground truth is written as a **header-only** `rois_mapping.txt` and B scores will be zero.

---

## interf1 — Metric A (workflow reasoning)

**Required file:** `interf1/ground_truth_CoT.json`

A JSON array — one entry per WSI case:

```json
[
  {
    "id": "PIT_01_00020_01.tiff",
    "chain-of-thought": [
      { "question": "What is the organ?", "answer": "Breast", "next_question": "Is there any abnormality present?" },
      { "question": "Is there any abnormality present?", "answer": "Yes, there is an abnormality.", "next_question": "..." },
      { "question": "What is the final pathology report?", "answer": "Breast, core needle biopsy; ...", "next_question": "" }
    ]
  }
]
```

Rules:
- `id` may include a file extension; it is stripped when building `manifest.json` and `test/input/` folder names.
- The last step in each chain must have `"next_question": ""`.
- Extra fields (e.g. `organ`) are dropped in the exported metric A file.

### WSI files

Optionally place the actual `.tiff` whole-slide image files in `interf1/` for your own reference. The prep scripts do **not** read WSI files — they read only `ground_truth_CoT.json`.

If interf1 is omitted entirely, metric A ground truth is written as an **empty JSON array** `[]` and A scores will be zero.
