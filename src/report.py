"""Final report synthesis after budget is exhausted."""

from typing import Any, Dict, List

from src.llm import LLMClient, chat


def generate_final_report(
    llm: LLMClient,
    user_query: str,
    iterations: List[Dict[str, Any]],
    temperature: float,
    max_paragraphs: int = 2,
) -> str:
    """Synthesize sub-question answers into a report grounded in the data lake."""
    if not iterations:
        return "No sub-questions were answered; unable to produce a report."

    blocks = []
    for i, it in enumerate(iterations, start=1):
        blocks.append(
            f"### Finding {i}\n"
            f"Sub-question: {it.get('sub_question')}\n"
            f"SQL Query: {it.get('sql')}\n"
            f"Answer: {it.get('answer')}\n"
        )
    evidence = "\n".join(blocks)

    prompt = f"""Your task is to answer the provided research question by synthesizing the provided grounded sub-analyses as evidence into a cohesive report.

Research Question:
{user_query}

Evidence:
{evidence}

Synthesize the evidence into a cohesive report (1-{max_paragraphs} paragraphs). Cite which table/analysis supports each claim. Do not invent facts beyond the evidence. Output ONLY the report. No other text.
"""
    return chat(
        llm,
        prompt,
        temperature=temperature,
        max_tokens=4096,
        feature="final_report",
    )
