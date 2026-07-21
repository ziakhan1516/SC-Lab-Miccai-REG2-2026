from pathlib import Path
from typing import Any, Dict, List, Optional


ID_KEYS = ("id", "sample_id", "case_id", "patient_id")
SLIDE_ID_KEYS = ("slide_id", "wsi_id", "wsi", "image_id", "svs_id", "file_name")
REPORT_KEYS = (
    "pathology_report",
    "final_report",
    "report",
    "diagnosis",
    "expert_report",
    "reference_report",
)
REASONING_KEYS = (
    "reasoning",
    "reasoning_flow",
    "rationale",
    "expert_reasoning",
    "structured_reasoning",
)
CHAIN_KEYS = ("chain-of-thought", "chain_of_thought", "cot", "reasoning_steps")


DEFAULT_INSTRUCTION = (
    "Given one whole-slide pathology image represented by learned visual "
    "features, generate the diagnostic workflow reasoning and the final "
    "pathology report. Use the exact canonical workflow question strings from "
    "the training annotations. Do not paraphrase workflow questions."
)


def _first_text(sample: Dict[str, Any], keys: tuple) -> str:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_id(sample: Dict[str, Any], keys: tuple) -> Optional[str]:
    for key in keys:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _clean_slide_id(value: str) -> str:
    return Path(str(value)).stem


def _chain_steps(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in CHAIN_KEYS:
        value = sample.get(key)
        if isinstance(value, list):
            return [step for step in value if isinstance(step, dict)]
    return []


def _is_report_question(question: str) -> bool:
    q = question.lower()
    return "final pathology report" in q or "pathology report" in q


def _format_reasoning_from_chain(steps: List[Dict[str, Any]]) -> str:
    lines = []
    reasoning_index = 1

    for step in steps:
        question = str(step.get("question", "")).strip()
        answer = str(step.get("answer", "")).strip()
        next_question = str(step.get("next_question", "")).strip()

        if not question and not answer:
            continue

        if question:
            lines.append(f"{reasoning_index}. Question: {question}")
        else:
            lines.append(f"{reasoning_index}. Question: Not specified")

        if answer:
            lines.append(f"   Answer: {answer}")
        if next_question:
            lines.append(f"   Next Question: {next_question}")

        reasoning_index += 1

    return "\n".join(lines).strip()


def _extract_report(sample: Dict[str, Any], steps: List[Dict[str, Any]]) -> str:
    direct_report = _first_text(sample, REPORT_KEYS)
    if direct_report:
        return direct_report

    for step in steps:
        question = str(step.get("question", "")).strip()
        answer = str(step.get("answer", "")).strip()
        if answer and _is_report_question(question):
            return answer

    if steps:
        last_answer = str(steps[-1].get("answer", "")).strip()
        if last_answer:
            return last_answer

    return ""


def build_generation_prompt(instruction: Optional[str] = None) -> str:
    task = instruction.strip() if instruction else DEFAULT_INSTRUCTION
    return (
        f"{task}\n\n"
        "Output format:\n"
        "Reasoning:\n"
        "1. Question: <canonical workflow question>\n"
        "   Answer: <diagnostic answer>\n"
        "   Next Question: <canonical next workflow question>\n"
        "...\n\n"
        "Pathology Report:\n"
        "<structured pathology report>\n\n"
        "Rules:\n"
        "- Keep workflow question text exactly canonical.\n"
        "- Use Answer, not Assessment, for each workflow step.\n"
        "- Include the final pathology report question in the reasoning chain "
        "when it exists.\n\n"
        "WSI features are provided as the visual input.\n\n"
        "Answer:\n"
    )


def build_target_text(reasoning: str, report: str) -> str:
    reasoning = reasoning.strip() or "No explicit reasoning reference was provided."
    report = report.strip() or "No structured pathology report reference was provided."
    return f"Reasoning:\n{reasoning}\n\nPathology Report:\n{report}"


def preprocess_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sample_id = _first_id(sample, ID_KEYS)
    slide_id = _first_id(sample, SLIDE_ID_KEYS) or sample_id

    if not sample_id or not slide_id:
        return None

    steps = _chain_steps(sample)
    reasoning = _format_reasoning_from_chain(steps) or _first_text(sample, REASONING_KEYS)
    report = _extract_report(sample, steps)

    if not reasoning and not report:
        return None

    instruction = str(sample.get("instruction", DEFAULT_INSTRUCTION)).strip()
    prompt = build_generation_prompt(instruction)
    target_text = build_target_text(reasoning, report)

    return {
        "id": sample_id,
        "slide_id": _clean_slide_id(slide_id),
        "instruction": instruction,
        "prompt": prompt,
        "reasoning": reasoning,
        "pathology_report": report,
        "target_text": target_text,
        "training_text": target_text,
        "workflow_steps": steps,
        "source": sample,
    }


def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_data = []

    for sample in data:
        processed = preprocess_sample(sample)
        if processed is not None:
            processed_data.append(processed)

    return processed_data
