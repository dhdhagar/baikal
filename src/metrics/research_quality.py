"""LLM rubric for deep-research finding and report quality."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.llm import LLMClient, chat, get_llm_client, parse_json_dict_from_llm
from src.metrics.common import (
    round4,
    snap_binary_score,
    snap_ordinal_score,
)
from src.metrics.rubric_utils import RUBRIC_COMPONENTS, extract_rubric_scores

ORDINAL_RUBRIC_GUIDE = """
For {dimension}, select exactly one label:
- "none" (0.0): {none_desc}
- "minimal" (0.25): {minimal_desc}
- "partial" (0.5): {partial_desc}
- "substantial" (0.75): {substantial_desc}
- "full" (1.0): {full_desc}
"""

ABSENCE_ONLY_RULE = """
Important — absence-only findings:
An answer is "absence-only" if it does NOT provide the factual information requested by the
sub-question, and instead only states that the information is missing, unavailable, not found,
not stated, not specified, or not present in the cited sources/SQL results. Examples:
- "Not stated in the provided documents."
- "No relevant information found."
- "The documents do not contain the peak chart position."

If the answer is absence-only, apply these overrides regardless of whether the absence claim
is technically supported by the cited evidence:
- grounded: "no" — the finding does not deliver substantively grounded information that answers the sub-question
- report_usefulness: "none" — documenting a search failure or data gap is not a useful report finding
- relevance and distinctness: score normally

Only score grounded "yes" when the answer provides the requested fact (e.g. a label, date, chart position,
certification level) supported by SQL and/or passage evidence.
"""

FINDING_RUBRIC_PROMPT = """Given a research question and a candidate finding on a data lake of tables and passages, your task is to evaluate the quality of the finding based on a provided rubric.

Research question:
"{user_query}"

Current finding:
Sub-question: {sub_question}
Answer: {answer}

Evidence for the finding:
Tables cited (SQL + answer): {tables_used}
Passages cited in answer: {passages_cited}
Cluster passages:
{cluster_passages}

Cited passage text:
{cited_passage_text}

SQL executed ({row_count} rows returned):
{sql}

SQL result preview:
{rows_preview}

Previously observed findings:
{prior_findings}

{absence_only_rule}

Evaluate the current finding on four rubric dimensions.

1) Grounded in table or passage evidence?
Select exactly one:
- "yes" (1.0): The answer provides the factual information requested by the sub-question, and that information is supported by the SQL results and/or cited passage text
- "no" (0.0): The answer is absence-only (see above), unsupported, contradicted by evidence, or not tied to available evidence

2) Relevance to the research question domain?
{relevance_guide}

3) Adds new information beyond previously observed findings?
{distinctness_guide}

4) Useful to include in a deep-research report (independent of other findings)?
{usefulness_guide}

Respond ONLY with a valid JSON:
{{
  "grounded_reasoning": "<one sentence>",
  "grounded": "yes" or "no",
  "relevance_reasoning": "<one sentence>",
  "relevance": "<one of: none, minimal, partial, substantial, full>",
  "distinctness_reasoning": "<one sentence>",
  "distinctness": "<one of: none, minimal, partial, substantial, full>",
  "report_usefulness_reasoning": "<one sentence>",
  "report_usefulness": "<one of: none, minimal, partial, substantial, full>"
}}
"""

RELEVANCE_GUIDE = ORDINAL_RUBRIC_GUIDE.format(
    dimension="relevance",
    none_desc="Unrelated or does not help answer the research question",
    minimal_desc="Barely related; mostly off-topic for the research goals",
    partial_desc="Tangentially related but misses main analytical goals",
    substantial_desc="Mostly relevant with minor gaps",
    full_desc="Directly addresses an important part of the research question",
)

DISTINCTNESS_GUIDE = ORDINAL_RUBRIC_GUIDE.format(
    dimension="distinctness",
    none_desc="Duplicate or near-verbatim rephrase of a prior finding",
    minimal_desc="Mostly repeats prior findings with trivial wording changes",
    partial_desc="Overlaps heavily with prior findings but adds a small new detail",
    substantial_desc="Mostly new angle or evidence with some overlap",
    full_desc="Clearly new insight not covered by prior findings",
)

USEFULNESS_GUIDE = ORDINAL_RUBRIC_GUIDE.format(
    dimension="report usefulness",
    none_desc="Not worth including; noise, redundant in a report, or absence-only (only states that information was not found)",
    minimal_desc="Marginally informative; unlikely to help the reader",
    partial_desc="Somewhat helpful but low priority for the report",
    substantial_desc="Useful — addresses requested aspects or adds complementary insight",
    full_desc="Highly useful — directly answers part of the query or adds valuable complementary insight",
)


def parse_judge_model_specs(
    judge_models: str,
    default_provider: str = "openai",
) -> List[tuple[str, str]]:
    """
    Parse comma-separated judge specs.

    Each entry may be ``model`` (uses ``default_provider``) or
    ``provider:model`` (e.g. ``openai:gpt-5-mini,litellm:azure/gpt-4o``).
    """
    from src.llm import SUPPORTED_LLM_PROVIDERS

    default_provider = (default_provider or "openai").strip().lower()
    specs: List[tuple[str, str]] = []
    for entry in (judge_models or "").split(","):
        value = entry.strip()
        if not value:
            continue
        if ":" in value:
            provider, model = value.split(":", 1)
            provider = provider.strip().lower()
            model = model.strip()
            if provider in SUPPORTED_LLM_PROVIDERS and model:
                specs.append((provider, model))
                continue
        specs.append((default_provider, value))
    return specs or [(default_provider, "gpt-5-mini")]


def build_judge_clients(default_provider: str, judge_models: str) -> List[LLMClient]:
    return [
        get_llm_client(provider, model)
        for provider, model in parse_judge_model_specs(judge_models, default_provider)
    ]


def _rows_preview(execution: Optional[dict], max_rows: int = 5) -> str:
    if not execution:
        return "(no SQL execution)"
    rows = execution.get("rows") or []
    if not rows:
        return "(no rows)"
    return str(rows[:max_rows])


def _format_prior_findings(prior_findings: List[Dict[str, str]]) -> str:
    if not prior_findings:
        return "(none)"
    blocks = []
    for i, finding in enumerate(prior_findings, start=1):
        blocks.append(
            f"{i}. Sub-question: {finding.get('sub_question', '')}\n"
            f"   Answer: {finding.get('answer', '')}"
        )
    return "\n".join(blocks)


def _format_cluster_passages(iteration: dict) -> str:
    cluster = iteration.get("selected_cluster") or {}
    passage_ids = cluster.get("passage_ids") or []
    lines: List[str] = []
    for item in passage_ids:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append(f"  [{item[0]}] {item[1]}")
        elif isinstance(item, (list, tuple)) and item:
            lines.append(f"  [{item[0]}]")
        elif isinstance(item, str):
            lines.append(f"  [{item}]")
    return "\n".join(lines) if lines else "(no cluster passages)"


CITED_PASSAGE_TEXT_MAX_CHARS = 2500


def _format_cited_passage_text(
    passage_ids: List[str],
    passage_descriptions: Optional[Dict[str, Any]],
    *,
    max_chars: int = CITED_PASSAGE_TEXT_MAX_CHARS,
) -> str:
    if not passage_ids:
        return "(none cited in answer)"
    if not passage_descriptions:
        return "(passage text unavailable)"
    lines: List[str] = []
    for pid in sorted(passage_ids):
        record = passage_descriptions.get(pid) or passage_descriptions.get(pid.upper())
        if not isinstance(record, dict):
            lines.append(f"  [{pid}] (text unavailable)")
            continue
        title = str(record.get("title") or "")
        text = str(record.get("text") or "")[:max_chars]
        lines.append(f"  [{pid}] {title}. {text}".rstrip())
    return "\n".join(lines)


def _aggregate_component_scores(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "variance": None, "values": []}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round4(float(np.mean(arr))),
        "variance": round4(float(np.var(arr))),
        "values": [round4(float(v)) for v in values],
    }


def _parse_judge_response(raw: str) -> Dict[str, Any]:
    parsed = parse_json_dict_from_llm(raw)
    return {
        "grounded": snap_binary_score(parsed.get("grounded")),
        "relevance": snap_ordinal_score(parsed.get("relevance")),
        "distinctness": snap_ordinal_score(parsed.get("distinctness")),
        "report_usefulness": snap_ordinal_score(parsed.get("report_usefulness")),
        "reasoning": {
            key: str(parsed.get(f"{key}_reasoning", "")) for key in RUBRIC_COMPONENTS
        },
    }


def _judge_finding_with_llm(
    llm: LLMClient,
    *,
    user_query: str,
    iteration: dict,
    prior_findings: List[Dict[str, str]],
    temperature: float,
    passage_descriptions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    execution = iteration.get("execution") or {}
    sql = iteration.get("sql") or "(none)"
    row_count = int(execution.get("row_count", 0) or 0)
    passages_cited = iteration.get("passages_cited") or []
    prompt = FINDING_RUBRIC_PROMPT.format(
        user_query=user_query,
        sub_question=iteration.get("sub_question") or "",
        answer=iteration.get("answer") or "",
        tables_used=", ".join(iteration.get("tables_used") or []) or "(none)",
        passages_cited=", ".join(passages_cited) or "(none)",
        cluster_passages=_format_cluster_passages(iteration),
        cited_passage_text=_format_cited_passage_text(
            list(passages_cited),
            passage_descriptions,
        ),
        sql=sql,
        row_count=row_count,
        rows_preview=_rows_preview(execution if iteration.get("needs_sql") else None),
        prior_findings=_format_prior_findings(prior_findings),
        absence_only_rule=ABSENCE_ONLY_RULE.strip(),
        relevance_guide=RELEVANCE_GUIDE.strip(),
        distinctness_guide=DISTINCTNESS_GUIDE.strip(),
        usefulness_guide=USEFULNESS_GUIDE.strip(),
    )
    raw = chat(
        llm,
        prompt,
        system_prompt="You are a skilled data analyst evaluating the quality of research findings on data lakes.",
        temperature=temperature,
        max_tokens=4096,
        feature="metrics_finding_rubric",
        json_mode=True,
    )
    try:
        return _parse_judge_response(raw)
    except ValueError as exc:
        return {
            "grounded": 0.0,
            "relevance": 0.0,
            "distinctness": 0.0,
            "report_usefulness": 0.0,
            "reasoning": {"error": f"parse error: {exc}"},
        }


def compute_finding_score(component_means: Dict[str, Optional[float]]) -> float:
    score = 1.0
    for key in RUBRIC_COMPONENTS:
        value = component_means.get(key)
        if value is None:
            return 0.0
        score *= float(value)
    return round4(score) or 0.0


def judge_finding_rubric(
    judge_llms: List[LLMClient],
    *,
    user_query: str,
    iteration: dict,
    prior_findings: List[Dict[str, str]],
    temperature: float = 1.0,
    on_detail: Optional[Callable[[str], None]] = None,
    passage_descriptions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score one finding with one or more judge LLMs."""
    if not judge_llms:
        return {
            "finding_score": 0.0,
            "components": {},
            "judges": [],
        }

    per_judge: List[Dict[str, Any]] = []
    component_values: Dict[str, List[float]] = {k: [] for k in RUBRIC_COMPONENTS}

    for llm in judge_llms:
        if on_detail:
            on_detail(f"judge {llm.provider}:{llm.model}")
        judge_result = _judge_finding_with_llm(
            llm,
            user_query=user_query,
            iteration=iteration,
            prior_findings=prior_findings,
            temperature=temperature,
            passage_descriptions=passage_descriptions,
        )
        per_judge.append(
            {
                "provider": llm.provider,
                "model": llm.model,
                "scores": {
                    key: round4(judge_result[key]) for key in RUBRIC_COMPONENTS
                },
                "reasoning": judge_result.get("reasoning") or {},
            }
        )
        for key in RUBRIC_COMPONENTS:
            component_values[key].append(float(judge_result[key]))

    components = {
        key: _aggregate_component_scores(component_values[key])
        for key in RUBRIC_COMPONENTS
    }
    component_means = {key: components[key]["mean"] for key in RUBRIC_COMPONENTS}
    finding_score = compute_finding_score(component_means)

    return {
        "finding_score": finding_score,
        "components": components,
        "judges": per_judge,
    }


def build_research_quality_step(
    *,
    is_finding: bool,
    rubric: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    finding_score = 0.0
    if is_finding and rubric is not None:
        finding_score = float(rubric.get("finding_score") or 0.0)
    return {
        "is_finding": is_finding,
        "finding_score": round4(finding_score),
        "rubric": rubric,
    }


def compute_rubric_means_from_per_step(
    per_step: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Average each rubric dimension across all budget steps (missing rubric -> 0)."""
    if not per_step:
        return {key: None for key in RUBRIC_COMPONENTS}
    accumulators: Dict[str, List[float]] = {key: [] for key in RUBRIC_COMPONENTS}
    for step_metrics in per_step:
        rubric = (step_metrics.get("research_quality") or {}).get("rubric") or {}
        scores = extract_rubric_scores(rubric)
        for key in RUBRIC_COMPONENTS:
            value = scores.get(key)
            accumulators[key].append(float(value or 0.0) if value is not None else 0.0)
    return {
        key: round4(sum(values) / len(values)) if values else None
        for key, values in accumulators.items()
    }


def build_research_quality_summary(
    *,
    finding_scores: List[float],
    budget: int,
    per_step: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    total = float(sum(finding_scores))
    report_score = total / budget if budget > 0 else 0.0
    summary: Dict[str, Any] = {
        "finding_scores_sum": round4(total),
        "report_score": round4(report_score),
        "budget": budget,
        "n_findings_valid": sum(1 for score in finding_scores if score > 0),
    }
    if per_step is not None:
        summary["rubric_means"] = compute_rubric_means_from_per_step(per_step)
    return summary
