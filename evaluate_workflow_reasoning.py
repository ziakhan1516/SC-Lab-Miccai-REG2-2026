#!/usr/bin/env python3
"""
Evaluate Workflow Reasoning as described in Workflow Reasoning.pdf.

Per-case score:
  0.05 * Binary Path Validity
  + 0.30 * Edge-F1
  + 0.25 * MESS
  + 0.40 * Final Report Score
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from json_loader import load_json
from metrics import bleu_score, rouge_l
from multimodal_alignment import load_generator
from text_preprocess import preprocess_data
from wsi_dataset import discover_wsi_files, load_wsi_features, normalize_slide_id


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "no", "not", "of", "on", "or", "the",
    "there", "to", "with", "yes",
}


def canonicalize(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return text.rstrip(" .,:;")


def is_final_report_question(question: str) -> bool:
    q = canonicalize(question)
    return "pathology report" in q or "final report" in q


def tokenize_keywords(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*", text.lower())
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in STOPWORDS
    }


def keyword_jaccard(predicted: str, reference: str) -> float:
    pred_keywords = tokenize_keywords(predicted)
    ref_keywords = tokenize_keywords(reference)

    if not pred_keywords and not ref_keywords:
        return 1.0
    if not pred_keywords or not ref_keywords:
        return 0.0

    return len(pred_keywords & ref_keywords) / len(pred_keywords | ref_keywords)


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: Optional[str] = None,
        disabled: bool = False,
    ):
        self.disabled = disabled
        self.cache: Dict[str, torch.Tensor] = {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        if disabled:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print(f"Loaded embedding model: {model_name}")
        except Exception as exc:
            print(f"Embedding model unavailable, using lexical fallback: {exc}")
            self.disabled = True

    def encode(self, text: str) -> torch.Tensor:
        key = str(text)
        if key in self.cache:
            return self.cache[key]

        if self.disabled or self.model is None or self.tokenizer is None:
            vec = self._lexical_vector(key)
            self.cache[key] = vec
            return vec

        with torch.no_grad():
            encoded = self.tokenizer(
                key,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            vec = torch.nn.functional.normalize(pooled.squeeze(0).cpu(), p=2, dim=0)

        self.cache[key] = vec
        return vec

    def cosine(self, predicted: str, reference: str) -> float:
        # Identical answers are maximally similar for any embedder; this also
        # keeps the lexical fallback meaningful for short/numeric answers.
        if str(predicted).strip() == str(reference).strip():
            return 1.0
        pred_vec = self.encode(predicted)
        ref_vec = self.encode(reference)
        return float(torch.dot(pred_vec, ref_vec).clamp(min=-1.0, max=1.0))

    @staticmethod
    def _lexical_vector(text: str) -> torch.Tensor:
        # Stable hashed bag-of-keywords fallback for machines without PubMedBERT.
        dim = 512
        vector = torch.zeros(dim)
        for keyword in tokenize_keywords(text):
            vector[hash(keyword) % dim] += 1.0
        if vector.norm() == 0:
            return vector
        return torch.nn.functional.normalize(vector, p=2, dim=0)


class WorkflowParser:
    # Line-oriented prefixes. A line-based parser round-trips the training
    # format losslessly (the old regex dropped "Next Question:" edges).
    _REPORT_LINE = re.compile(r"^\s*(?:final\s+)?pathology\s+report\s*:", re.IGNORECASE)
    _NEXT_LINE = re.compile(r"^\s*next\s+question\s*:\s*(?P<next>.*)$", re.IGNORECASE)
    _Q_LINE = re.compile(
        r"^\s*(?:\d+[\.\)]\s*)?(?:diagnostic\s+)?question\s*:\s*(?P<question>.*)$",
        re.IGNORECASE,
    )
    _A_LINE = re.compile(
        r"^\s*(?:answer|assessment|response)\s*:\s*(?P<answer>.*)$",
        re.IGNORECASE,
    )

    @classmethod
    def parse_steps(cls, text: str) -> List[Dict[str, str]]:
        steps: List[Dict[str, str]] = []
        current: Optional[Dict[str, str]] = None
        field: Optional[str] = None  # 'question' | 'answer' | 'next_question'

        for raw_line in str(text).splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # The pathology-report section ends the reasoning chain.
            if cls._REPORT_LINE.match(line):
                break

            next_match = cls._NEXT_LINE.match(line)
            if next_match:
                if current is not None:
                    current["next_question"] = next_match.group("next").strip()
                    field = "next_question"
                continue

            q_match = cls._Q_LINE.match(line)
            if q_match:
                if current is not None:
                    steps.append(current)
                current = {
                    "question": q_match.group("question").strip(),
                    "answer": "",
                    "next_question": "",
                }
                field = "question"
                continue

            a_match = cls._A_LINE.match(line)
            if a_match:
                if current is not None:
                    current["answer"] = a_match.group("answer").strip()
                    field = "answer"
                continue

            # Continuation line: extend whichever field is currently open.
            if current is not None and field is not None:
                current[field] = (current[field] + " " + line).strip()

        if current is not None:
            steps.append(current)

        cleaned = []
        for step in steps:
            step = {k: cls._clean_field(v) for k, v in step.items()}
            if step["question"]:
                cleaned.append(step)
        return cleaned

    @classmethod
    def parse_cot(cls, text: str) -> List[Dict[str, str]]:
        return cls.parse_steps(text)

    @staticmethod
    def _clean_field(text: str) -> str:
        text = re.sub(r"\s+", " ", str(text).strip())
        return text.rstrip()

    @classmethod
    def extract_report(cls, text: str, steps: Optional[List[Dict[str, str]]] = None) -> str:
        match = re.search(r"Pathology\s+Report:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        for step in steps or cls.parse_steps(text):
            if is_final_report_question(step.get("question", "")):
                return step.get("answer", "").strip()

        return ""

    @staticmethod
    def edges(steps: Iterable[Dict[str, str]]) -> Set[Tuple[str, str]]:
        result = set()
        for step in steps:
            question = canonicalize(step.get("question", ""))
            next_question = canonicalize(step.get("next_question", ""))
            if question and next_question:
                result.add((question, next_question))
        return result

    @staticmethod
    def extract_edges(steps: Iterable[Dict[str, str]]) -> Set[Tuple[str, str]]:
        return WorkflowParser.edges(steps)


ChainOfThoughtParser = WorkflowParser


class WorkflowReasoningMetrics:
    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        embedding_model: Optional[str] = None,
    ):
        self.embedder = embedder or TextEmbedder(
            model_name=embedding_model
            or "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            disabled=embedding_model is None,
        )

    @staticmethod
    def binary_path_validity(pred_edges: Set[Tuple[str, str]], gt_edges: Set[Tuple[str, str]]) -> float:
        return 1.0 if pred_edges == gt_edges else 0.0

    @staticmethod
    def edge_f1(pred_edges: Set[Tuple[str, str]], gt_edges: Set[Tuple[str, str]]) -> Dict[str, float]:
        if not pred_edges and not gt_edges:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        tp = len(pred_edges & gt_edges)
        fp = len(pred_edges - gt_edges)
        fn = len(gt_edges - pred_edges)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def mess(self, pred_steps: List[Dict[str, str]], gt_steps: List[Dict[str, str]]) -> float:
        pred_by_edge = {
            (canonicalize(step["question"]), canonicalize(step.get("next_question", ""))): step.get("answer", "")
            for step in pred_steps
        }

        scores = []
        for gt_step in gt_steps:
            if is_final_report_question(gt_step.get("question", "")):
                continue

            edge = (
                canonicalize(gt_step.get("question", "")),
                canonicalize(gt_step.get("next_question", "")),
            )
            if not edge[0] or not edge[1]:
                continue

            if edge not in pred_by_edge:
                scores.append(0.0)
                continue

            similarity = self.embedder.cosine(pred_by_edge[edge], gt_step.get("answer", ""))
            scores.append(max(0.0, min(1.0, similarity)))

        return sum(scores) / len(scores) if scores else 0.0

    def mean_edge_semantic_similarity(
        self,
        pred_steps: List[Dict[str, str]],
        gt_steps: List[Dict[str, str]],
        pred_edges: Optional[Set[Tuple[str, str]]] = None,
        gt_edges: Optional[Set[Tuple[str, str]]] = None,
    ) -> float:
        return self.mess(pred_steps, gt_steps)

    @staticmethod
    def workflow_reasoning_score(
        bpv: float,
        edge_f1: float,
        mess: float,
        final_report_score: float = 0.0,
    ) -> float:
        return 0.05 * bpv + 0.30 * edge_f1 + 0.25 * mess + 0.40 * final_report_score

    def final_report_score(self, predicted_report: str, reference_report: str) -> Dict[str, float]:
        if not predicted_report and not reference_report:
            return {
                "score": 1.0,
                "rouge_l_f1": 1.0,
                "bleu_4": 1.0,
                "keyword_jaccard": 1.0,
                "embedding_cosine_rescaled": 1.0,
            }

        rouge_l_f1 = rouge_l(predicted_report, reference_report)["f1"] / 100.0
        bleu_4 = bleu_score(predicted_report, reference_report)["bleu_4"] / 100.0
        keywords = keyword_jaccard(predicted_report, reference_report)
        cosine = self.embedder.cosine(predicted_report, reference_report)
        cosine_rescaled = max(0.0, min(1.0, (cosine - 0.5) / 0.5))

        score = (
            0.15 * (rouge_l_f1 + bleu_4)
            + 0.40 * keywords
            + 0.30 * cosine_rescaled
        )

        return {
            "score": max(0.0, min(1.0, score)),
            "rouge_l_f1": rouge_l_f1,
            "bleu_4": bleu_4,
            "keyword_jaccard": keywords,
            "embedding_cosine_rescaled": cosine_rescaled,
        }

    def score_case(
        self,
        generated: str,
        reference: str,
        gt_steps: Optional[List[Dict[str, str]]] = None,
        gt_report: Optional[str] = None,
    ) -> Dict[str, object]:
        pred_steps = WorkflowParser.parse_steps(generated)
        # Prefer the raw ground-truth chain-of-thought (exact canonical edges)
        # over re-parsing our own formatted reference text.
        if gt_steps is None:
            gt_steps = WorkflowParser.parse_steps(reference)
        pred_edges = WorkflowParser.edges(pred_steps)
        gt_edges = WorkflowParser.edges(gt_steps)

        if gt_report is None:
            gt_report = WorkflowParser.extract_report(reference, gt_steps)

        bpv = self.binary_path_validity(pred_edges, gt_edges)
        edge = self.edge_f1(pred_edges, gt_edges)
        mess = self.mess(pred_steps, gt_steps)
        final_report = self.final_report_score(
            WorkflowParser.extract_report(generated, pred_steps),
            gt_report,
        )

        workflow_score = (
            0.05 * bpv
            + 0.30 * edge["f1"]
            + 0.25 * mess
            + 0.40 * final_report["score"]
        )

        return {
            "binary_path_validity": bpv,
            "edge_f1": edge,
            "mess": mess,
            "final_report_score": final_report,
            "workflow_reasoning_score": workflow_score,
            "predicted_steps": pred_steps,
            "ground_truth_steps": gt_steps,
            "predicted_edges": sorted(list(pred_edges)),
            "ground_truth_edges": sorted(list(gt_edges)),
        }


def ground_truth_from_sample(
    sample: dict,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """Return (gt_steps, gt_report) from the raw chain-of-thought if available.

    Falling back to (None, None) lets the scorer re-parse the formatted
    reference text instead.
    """
    raw_steps = sample.get("workflow_steps") or []
    if not raw_steps:
        return None, None

    gt_report = ""
    for step in raw_steps:
        if is_final_report_question(step.get("question", "")):
            gt_report = str(step.get("answer", "")).strip()
            break
    if not gt_report:
        gt_report = str(raw_steps[-1].get("answer", "")).strip()

    return raw_steps, gt_report


def _report_of_chain(cot: List[Dict[str, str]]) -> str:
    for step in cot:
        if is_final_report_question(step.get("question", "")):
            return str(step.get("answer", "")).strip()
    return str(cot[-1].get("answer", "")).strip() if cot else ""


def score_chains(scorer, pred_cot, gt_cot) -> Dict[str, float]:
    """Score a predicted chain-of-thought (list of step dicts) against a
    ground-truth chain using the Workflow Reasoning metric components."""
    pred_edges = WorkflowParser.edges(pred_cot)
    gt_edges = WorkflowParser.edges(gt_cot)
    bpv = scorer.binary_path_validity(pred_edges, gt_edges)
    edge_f1 = scorer.edge_f1(pred_edges, gt_edges)["f1"]
    mess = scorer.mess(pred_cot, gt_cot)
    final = scorer.final_report_score(_report_of_chain(pred_cot), _report_of_chain(gt_cot))["score"]
    workflow = 0.05 * bpv + 0.30 * edge_f1 + 0.25 * mess + 0.40 * final
    return {
        "binary_path_validity": bpv,
        "edge_f1": edge_f1,
        "mess": mess,
        "final_report_score": final,
        "workflow_reasoning_score": workflow,
    }


def is_processed_dataset(records: List[dict]) -> bool:
    return bool(records) and "target_text" in records[0] and "slide_id" in records[0]


def load_dataset(json_path: str, root_key: Optional[str]) -> List[dict]:
    records = load_json(json_path, root_key=root_key)
    return records if is_processed_dataset(records) else preprocess_data(records)


def split_dataset(samples: List[dict], test_size: float, seed: int) -> Tuple[List[dict], List[dict]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_size)))
    return shuffled[test_count:], shuffled[:test_count]


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: List[dict]) -> Dict[str, float]:
    metrics = [row["metrics"] for row in results]
    return {
        "binary_path_validity": mean([m["binary_path_validity"] for m in metrics]),
        "edge_f1": mean([m["edge_f1"]["f1"] for m in metrics]),
        "mess": mean([m["mess"] for m in metrics]),
        "final_report_score": mean([m["final_report_score"]["score"] for m in metrics]),
        "workflow_reasoning_score": mean([m["workflow_reasoning_score"] for m in metrics]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Workflow Reasoning PDF metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--json", required=True, help="Raw JSON or saved split JSON.")
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--root-key", default=None)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        "--num-samples",
        dest="limit",
        type=int,
        default=None,
        help="Evaluate on this many cases (randomly sampled unless --no-shuffle).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Take the first --limit cases in file order instead of random.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--embedding-model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--disable-embeddings", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--output", default="outputs/workflow_reasoning_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using generation device: {device}")

    samples = load_dataset(args.json, args.root_key)
    file_map = discover_wsi_files(args.pt_dir)
    samples = [
        sample
        for sample in samples
        if normalize_slide_id(sample["slide_id"]) in file_map
    ]

    if not is_processed_dataset(load_json(args.json, root_key=args.root_key)):
        _, samples = split_dataset(samples, test_size=args.test_size, seed=args.seed)

    if args.limit:
        if args.no_shuffle:
            samples = samples[:args.limit]
        else:
            shuffled = list(samples)
            random.Random(args.seed).shuffle(shuffled)
            samples = shuffled[:args.limit]

    if not samples:
        raise ValueError("No matched evaluation samples found.")

    print(f"Evaluation samples: {len(samples)}")
    model = load_generator(args.checkpoint, device=device)
    embedder = TextEmbedder(
        model_name=args.embedding_model,
        device=args.embedding_device or device,
        disabled=args.disable_embeddings,
    )
    scorer = WorkflowReasoningMetrics(embedder)

    results = []
    for idx, sample in enumerate(samples, start=1):
        slide_id = normalize_slide_id(sample["slide_id"])
        features = load_wsi_features(str(file_map[slide_id]))

        generated = model.generate(
            features=[features],
            prompts=[sample["prompt"]],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )[0].strip()

        gt_steps, gt_report = ground_truth_from_sample(sample)
        case_metrics = scorer.score_case(
            generated,
            sample["target_text"],
            gt_steps=gt_steps,
            gt_report=gt_report,
        )
        results.append(
            {
                "id": sample["id"],
                "slide_id": slide_id,
                "generated_text": generated,
                "reference_text": sample["target_text"],
                "metrics": case_metrics,
            }
        )

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(samples)}] {slide_id}")
        print(f"{'-'*80}\n--- GENERATED ---\n{generated}")
        print(f"\n--- REFERENCE ---\n{sample['target_text']}")
        print(
            f"\n--- SCORES ---  "
            f"BPV={case_metrics['binary_path_validity']:.3f} "
            f"EdgeF1={case_metrics['edge_f1']['f1']:.3f} "
            f"MESS={case_metrics['mess']:.3f} "
            f"FinalReport={case_metrics['final_report_score']['score']:.3f} "
            f"WorkflowReasoning={case_metrics['workflow_reasoning_score']:.3f}"
        )

    summary = summarize(results)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    payload = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "json": args.json,
            "pt_dir": args.pt_dir,
            "num_samples": len(results),
            "test_size": args.test_size,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "embedding_model": None if args.disable_embeddings else args.embedding_model,
            "weights": {
                "binary_path_validity": 0.05,
                "edge_f1": 0.30,
                "mess": 0.25,
                "final_report_score": 0.40,
            },
        },
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation output: {output_path}")


if __name__ == "__main__":
    main()
