"""Report-generation metrics on the held-out test split.

Computes, on the *pathology report* the model generates versus the reference
report, the standard text-generation scores used in report-generation papers:

    BLEU-1, BLEU-2, BLEU-3, BLEU-4   (corpus-level, NLTK, method-1 smoothing)
    ROUGE-L  F1                      (rouge_score, stemmed)
    BERTScore F1                     (bert_score, roberta-large by default)
    GLEU  ("green" score)            (corpus-level Google-BLEU, NLTK)

It reuses exactly the same model loading, test split, WSI features, and report
parser as ``evaluate_test_set.py`` / the in-training eval, so the reports being
scored are identical to those behind the Workflow-Reasoning numbers.

Two ways to run
---------------
1. Load the model and generate (default -- matches evaluate_test_set.py):

    python evaluate_report_metrics.py \
        --checkpoint checkpoints/wsi_reasoning_r1qwen1p5b_conch

2. Reuse generations already saved by evaluate_test_set.py (fast, no GPU):

    python evaluate_report_metrics.py \
        --checkpoint checkpoints/wsi_reasoning_r1qwen1p5b_conch \
        --from-json  checkpoints/wsi_reasoning_r1qwen1p5b_conch/test_eval_v3.json

Run with the peft-0.17 environment (``manga``/``abdul``); the default py3.8
env cannot deserialize the LoRA adapter. See the project env notes.
"""

import argparse
import json
import re
from pathlib import Path

# --- Tokenisation shared by BLEU / GLEU / ROUGE-fallback ---------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    return _TOKEN_RE.findall(str(text).lower())


# =============================================================================
#  Metric computation (library-backed, corpus level where standard)
# =============================================================================
def compute_metrics(pairs, bertscore_model=None, no_bertscore=False,
                    bertscore_baseline=False, device=None):
    """pairs: list of (hypothesis_report, reference_report) raw strings."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.gleu_score import corpus_gleu
    from rouge_score import rouge_scorer

    # Drop pairs with an empty reference (nothing to score against); an empty
    # hypothesis is kept and scored as (near-)zero so it is not silently ignored.
    hyps_raw, refs_raw = [], []
    n_empty_ref = 0
    for hyp, ref in pairs:
        if not str(ref).strip():
            n_empty_ref += 1
            continue
        hyps_raw.append(str(hyp))
        refs_raw.append(str(ref))

    n = len(hyps_raw)
    if n == 0:
        raise SystemExit("No scorable pairs (all references empty).")

    hyp_tok = [tokenize(h) or ["<empty>"] for h in hyps_raw]
    ref_tok = [tokenize(r) for r in refs_raw]
    list_of_refs = [[r] for r in ref_tok]  # single reference per hypothesis

    smooth = SmoothingFunction().method1
    weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }
    scores = {
        name: corpus_bleu(list_of_refs, hyp_tok, weights=w,
                          smoothing_function=smooth)
        for name, w in weights.items()
    }

    # GLEU ("green"): corpus-level Google-BLEU (n=1..4).
    scores["gleu"] = corpus_gleu(list_of_refs, hyp_tok)

    # ROUGE-L F1: mean of per-report F1 (stemmed), the usual convention.
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_f1 = sum(
        rscorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hyps_raw, refs_raw)
    ) / n
    scores["rouge_l_f1"] = rouge_l_f1

    # BERTScore F1 (contextual embedding overlap).
    if not no_bertscore:
        from bert_score import score as bertscore_score
        kw = dict(cands=hyps_raw, refs=refs_raw,
                  rescale_with_baseline=bertscore_baseline,
                  verbose=True)
        if bertscore_model:
            kw["model_type"] = bertscore_model
        else:
            kw["lang"] = "en"  # -> roberta-large
        if device:
            kw["device"] = device
        _, _, F = bertscore_score(**kw)
        scores["bertscore_f1"] = float(F.mean())
    else:
        scores["bertscore_f1"] = None

    scores["_num_scored"] = n
    scores["_num_empty_ref_skipped"] = n_empty_ref
    return scores


# =============================================================================
#  Obtaining (hypothesis, reference) report pairs
# =============================================================================
def pairs_from_json(json_path, field):
    """Reuse generations saved by evaluate_test_set.py."""
    from evaluate_workflow_reasoning import WorkflowParser

    with open(json_path) as f:
        payload = json.load(f)
    results = payload["results"] if isinstance(payload, dict) else payload

    pairs = []
    for r in results:
        gen = r["generated_text"]
        ref = r["reference_text"]
        if field == "report":
            gen = WorkflowParser.extract_report(gen)
            ref = WorkflowParser.extract_report(ref)
        pairs.append((gen, ref))
    return pairs


def pairs_from_model(args, field, device):
    """Load the trained model and generate on the test split (like
    evaluate_test_set.py)."""
    import torch
    from multimodal_alignment import load_generator
    from wsi_dataset import (
        load_wsi_features, discover_wsi_files, normalize_slide_id,
    )
    from evaluate_workflow_reasoning import WorkflowParser

    ckpt = Path(args.checkpoint)
    test_json = Path(args.test_json) if args.test_json else ckpt / "splits" / "test.json"
    if not test_json.exists():
        raise FileNotFoundError(f"Test split not found: {test_json}. Pass --test-json.")

    pt_dir = args.pt_dir
    if pt_dir is None:
        targs = ckpt / "training_args.json"
        if targs.exists():
            with targs.open() as f:
                pt_dir = json.load(f).get("pt_dir")
    if not pt_dir:
        raise ValueError("Could not resolve --pt-dir (no training_args.json). Pass it.")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test split : {test_json}")
    print(f"WSI pt dir : {pt_dir}")
    print(f"Device     : {device}\n")

    with open(test_json) as f:
        test_data = json.load(f)
    file_map = discover_wsi_files(pt_dir)

    print("Loading model ...")
    model = load_generator(args.checkpoint, device=device)
    model.eval()

    total = len(test_data) if args.limit <= 0 else min(args.limit, len(test_data))
    print(f"Generating on {total} test cases ...\n")

    pairs, per_case, skipped = [], [], 0
    for idx, sample in enumerate(test_data, start=1):
        if args.limit and idx > args.limit:
            break
        slide_id = normalize_slide_id(sample["slide_id"])
        file_path = file_map.get(slide_id)
        if file_path is None:
            skipped += 1
            print(f"[{idx}/{total}] {slide_id}: no WSI feature file -> skipped")
            continue

        features = load_wsi_features(str(file_path))
        with torch.no_grad():
            generated = model.generate(
                features=[features],
                prompts=[sample["prompt"]],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )[0].strip()

        reference = sample["target_text"]
        gen = generated
        ref = reference
        if field == "report":
            gen = WorkflowParser.extract_report(generated)
            ref = WorkflowParser.extract_report(reference)
        pairs.append((gen, ref))
        per_case.append({
            "slide_id": slide_id,
            "generated_text": generated,
            "reference_text": reference,
        })
        print(f"[{idx}/{total}] {slide_id}  report_chars(gen/ref)={len(gen)}/{len(ref)}")

    print(f"\n{skipped} cases skipped (no features).")
    return pairs, per_case


# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Report-generation metrics on the test split.")
    p.add_argument("--checkpoint",
                   default="checkpoints/wsi_reasoning_r1qwen1p5b_conch",
                   help="Directory of the saved SFT model.")
    p.add_argument("--from-json", default=None,
                   help="Reuse generations from an evaluate_test_set.py report "
                        "(e.g. <ckpt>/test_eval_v3.json). Skips model loading.")
    p.add_argument("--field", choices=["report", "full"], default="report",
                   help="Score the extracted pathology report (default) or the "
                        "entire generated text.")
    p.add_argument("--test-json", default=None, help="Override path to test split json.")
    p.add_argument("--pt-dir", default=None, help="Override WSI .pt feature directory.")
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (auto if omitted).")
    p.add_argument("--limit", type=int, default=0, help="Only score first N cases (0 = all).")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--bertscore-model", default=None,
                   help="HF model for BERTScore (default: roberta-large via lang=en). "
                        "e.g. microsoft/deberta-xlarge-mnli, or a PubMedBERT id.")
    p.add_argument("--bertscore-baseline", action="store_true",
                   help="Rescale BERTScore with baseline (needs baseline files).")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Skip BERTScore (avoids the roberta-large download).")
    p.add_argument("--output", default=None,
                   help="Where to write the JSON report "
                        "(default: <checkpoint>/report_metrics.json).")
    args = p.parse_args()

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if args.from_json:
        print(f"Reading saved generations: {args.from_json}")
        print(f"Scoring field            : {args.field}\n")
        pairs = pairs_from_json(args.from_json, args.field)
        per_case = None
    else:
        pairs, per_case = pairs_from_model(args, args.field, device)

    print(f"\nComputing metrics on {len(pairs)} report pairs "
          f"(field={args.field}) ...")
    scores = compute_metrics(
        pairs,
        bertscore_model=args.bertscore_model,
        no_bertscore=args.no_bertscore,
        bertscore_baseline=args.bertscore_baseline,
        device=device,
    )

    order = ["bleu_1", "bleu_2", "bleu_3", "bleu_4",
             "rouge_l_f1", "bertscore_f1", "gleu"]
    labels = {
        "bleu_1": "BLEU-1", "bleu_2": "BLEU-2", "bleu_3": "BLEU-3",
        "bleu_4": "BLEU-4", "rouge_l_f1": "ROUGE-L F1",
        "bertscore_f1": "BERTScore F1", "gleu": "GLEU (green)",
    }

    print(f"\n{'='*52}")
    print(f"REPORT-GENERATION METRICS   ({scores['_num_scored']} reports, "
          f"{scores['_num_empty_ref_skipped']} empty-ref skipped)")
    print(f"field = {args.field}")
    print(f"{'='*52}")
    for k in order:
        v = scores[k]
        print(f"  {labels[k]:<14s} {'n/a' if v is None else f'{v:.4f}'}")
    print(f"{'='*52}")

    # LaTeX-ready single row (values x100, common in report-gen tables).
    def pct(k):
        return "--" if scores[k] is None else f"{100*scores[k]:.2f}"
    print("\nLaTeX row (values x100):")
    print("  " + " & ".join(pct(k) for k in order) + r" \\")

    out = Path(args.output) if args.output else Path(args.checkpoint) / "report_metrics.json"
    payload = {
        "checkpoint": args.checkpoint,
        "source": args.from_json or "model-generated",
        "field": args.field,
        "bertscore_model": args.bertscore_model or ("skipped" if args.no_bertscore else "roberta-large"),
        "scores": {k: scores[k] for k in order},
        "num_scored": scores["_num_scored"],
        "num_empty_ref_skipped": scores["_num_empty_ref_skipped"],
    }
    if per_case is not None:
        payload["results"] = per_case
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
