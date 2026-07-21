"""Parse the generation model's text into chain-of-thought steps.

Self-contained copy of the line-oriented parser used by the training-time metric
(evaluate_workflow_reasoning.WorkflowParser.parse_steps), so the container does
not depend on the full evaluation module. The model emits:

    <think>
    1. Question: ...
       Answer: ...
       Next Question: ...
    ...
    </think>
    Pathology Report:
    ...

which maps to a list of {question, answer, next_question}.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

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


# The official evaluator scores the report as the answer of the step whose
# question canonicalises to this string (evaluate_metrics.FINAL_REPORT_QUESTION).
FINAL_REPORT_QUESTION = "What is the final pathology report?"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).rstrip()


def _extract_trailing_report(text: str) -> str:
    """Text after a 'Pathology Report:' heading (the model's report block)."""
    m = re.search(r"pathology\s+report\s*:\s*(.*)", str(text),
                  flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    report = m.group(1)
    report = report.replace("</think>", " ").strip()
    return _clean(report)


def parse_cot_with_report(text: str) -> List[Dict[str, str]]:
    """Steps INCLUDING a final 'What is the final pathology report?' step.

    The trained model usually emits that step itself (kept by parse_cot_steps).
    This adds a safety net: if it is missing but a trailing 'Pathology Report:'
    block exists, append it as the canonical final-report step so the evaluator's
    get_final_report_answer() can find the report.
    """
    steps = parse_cot_steps(text)

    has_final = any(
        s["question"].strip().lower().rstrip("?") == "what is the final pathology report"
        for s in steps
    )
    if not has_final:
        report = _extract_trailing_report(text)
        if report:
            if steps:
                steps[-1]["next_question"] = FINAL_REPORT_QUESTION
            steps.append({
                "question": FINAL_REPORT_QUESTION,
                "answer": report,
                "next_question": "",
            })
    if steps:
        steps[-1]["next_question"] = ""
    return steps


def parse_cot_steps(text: str) -> List[Dict[str, str]]:
    # Drop the <think> wrapper if present; steps live inside it.
    text = str(text).replace("<think>", "\n").replace("</think>", "\n")

    steps: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    field: Optional[str] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _REPORT_LINE.match(line):
            break

        m = _NEXT_LINE.match(line)
        if m:
            if current is not None:
                current["next_question"] = m.group("next").strip()
                field = "next_question"
            continue

        m = _Q_LINE.match(line)
        if m:
            if current is not None:
                steps.append(current)
            current = {"question": m.group("question").strip(), "answer": "", "next_question": ""}
            field = "question"
            continue

        m = _A_LINE.match(line)
        if m:
            if current is not None:
                current["answer"] = m.group("answer").strip()
                field = "answer"
            continue

        if current is not None and field is not None:
            current[field] = (current[field] + " " + line).strip()

    if current is not None:
        steps.append(current)

    cleaned = []
    for s in steps:
        s = {k: _clean(v) for k, v in s.items()}
        if s["question"]:
            cleaned.append(s)
    if cleaned:
        cleaned[-1]["next_question"] = ""
    return cleaned
