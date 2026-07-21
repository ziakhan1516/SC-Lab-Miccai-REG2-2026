import math
import re
from collections import Counter
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(candidate: List[str], reference: List[str], n: int) -> float:
    cand_counts = _ngrams(candidate, n)
    ref_counts = _ngrams(reference, n)

    if not cand_counts:
        return 0.0

    overlap = 0
    for gram, count in cand_counts.items():
        overlap += min(count, ref_counts.get(gram, 0))

    return overlap / sum(cand_counts.values())


def bleu_score(candidate_text: str, reference_text: str, max_n: int = 4) -> Dict[str, float]:
    candidate = tokenize(candidate_text)
    reference = tokenize(reference_text)

    if not candidate or not reference:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    brevity_penalty = 1.0
    if len(candidate) < len(reference):
        brevity_penalty = math.exp(1 - len(reference) / max(1, len(candidate)))

    scores = {}
    for n in range(1, max_n + 1):
        precisions = []
        for order in range(1, n + 1):
            precision = _modified_precision(candidate, reference, order)
            if precision == 0.0:
                precision = 1.0 / (2 * max(1, len(candidate)))
            precisions.append(precision)

        log_precision = sum(math.log(p) for p in precisions) / n
        scores[f"bleu_{n}"] = 100.0 * brevity_penalty * math.exp(log_precision)

    return scores


def _prf(overlap: int, candidate_total: int, reference_total: int) -> Dict[str, float]:
    precision = overlap / candidate_total if candidate_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
    }


def rouge_n(candidate_text: str, reference_text: str, n: int) -> Dict[str, float]:
    candidate = tokenize(candidate_text)
    reference = tokenize(reference_text)
    cand_counts = _ngrams(candidate, n)
    ref_counts = _ngrams(reference, n)

    overlap = 0
    for gram, count in cand_counts.items():
        overlap += min(count, ref_counts.get(gram, 0))

    return _prf(overlap, sum(cand_counts.values()), sum(ref_counts.values()))


def _lcs_length(a: List[str], b: List[str]) -> int:
    previous = [0] * (len(b) + 1)

    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current

    return previous[-1]


def rouge_l(candidate_text: str, reference_text: str) -> Dict[str, float]:
    candidate = tokenize(candidate_text)
    reference = tokenize(reference_text)
    overlap = _lcs_length(candidate, reference)
    return _prf(overlap, len(candidate), len(reference))


def evaluate_generation(candidate_text: str, reference_text: str) -> Dict[str, Dict[str, float]]:
    return {
        "bleu": bleu_score(candidate_text, reference_text),
        "rouge_1": rouge_n(candidate_text, reference_text, 1),
        "rouge_2": rouge_n(candidate_text, reference_text, 2),
        "rouge_l": rouge_l(candidate_text, reference_text),
    }


def average_metrics(rows: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}

    totals: Dict[str, Dict[str, float]] = {}
    for row in rows:
        for metric_name, values in row.items():
            totals.setdefault(metric_name, {})
            for key, value in values.items():
                totals[metric_name][key] = totals[metric_name].get(key, 0.0) + value

    for metric_name, values in totals.items():
        for key in values:
            values[key] /= len(rows)

    return totals
