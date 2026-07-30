"""Sub-question generation, selection, and de-duplication."""

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Set

from src.llm import LLMClient, chat, parse_json_dict_from_llm
from src.utils import log


@dataclass(frozen=True)
class SubquestionGenerationResult:
    subquestions: List[str]
    parse_failed: bool = False

SUBQUESTION_SELECTION_RUBRIC = """
Pick the sub-question most likely to produce a high-quality report finding:

1) Grounded — answerable from the selected cluster's tables/passages (not speculative).
2) Relevance — directly helps answer the research question.
3) Distinctness — adds new information beyond sub-questions already asked.
4) Report usefulness — worth including in a deep-research report (not redundant noise).
"""


def format_cluster_context(
    cluster: dict,
    max_tables: Optional[int] = None,
    *,
    include_passages: bool = True,
) -> str:
    """Human-readable cluster blurb for prompts."""
    desc = cluster.get("description", "")
    lines = [f"Topics/descriptions: {desc}", "Tables:"]
    tables = cluster.get("tables", [])
    for t in tables[:max_tables]:
        cols = ", ".join((t.get("columns") or [])[:8])
        title = t.get("title", "")
        lines.append(
            f"- {t.get('table_id')}: {title}. Columns: {cols}. "
            f"Description: {(t.get('description') or '')[:400]}"
        )
    if max_tables:
        extra = len(tables) - max_tables
        if extra > 0:
            lines.append(f"... and {extra} more tables")
    if not tables:
        lines.append("(none)")

    passages = cluster.get("passages", []) if include_passages else []
    if passages:
        lines.append("Passages:")
        max_passages = max_tables if max_tables is not None else len(passages)
        for p in passages[:max_passages]:
            title = p.get("title", "")
            text_preview = (p.get("text") or "")[:300]
            lines.append(
                f"- {p.get('passage_id')}: {title}. {text_preview}"
            )
        if max_passages and len(passages) > max_passages:
            lines.append(f"... and {len(passages) - max_passages} more passages")

    return "\n".join(lines)


def _parse_remove_indices(raw_indices: Any) -> Optional[Set[int]]:
    """Parse 1-based candidate indices to remove; None if the field is invalid."""
    if raw_indices is None:
        return set()
    if not isinstance(raw_indices, list):
        return None
    remove: Set[int] = set()
    for item in raw_indices:
        try:
            remove.add(int(item))
        except (TypeError, ValueError):
            continue
    return remove


def generate_subquestions(
    llm: LLMClient,
    user_query: str,
    cluster: dict,
    k: int,
    temperature: float,
    previous_subquestions: Optional[List[str]] = None,
    *,
    allow_empty: bool = True,
    silent: bool = True,
) -> SubquestionGenerationResult:
    """Ask LLM for diverse sub-questions answerable from this cluster."""
    has_passages = bool(cluster.get("passages"))
    has_tables = bool(cluster.get("tables"))
    if has_tables and has_passages:
        source_note = (
            "available tables (via SQL) and passages (text summaries)"
        )
        answerable_note = (
            "Each sub-question must be answerable using the available tables, passages, "
            "or both. Some may require SQL over tables; others may be answerable from "
            "passages alone."
        )
    elif has_passages:
        source_note = "available passages (text summaries)"
        answerable_note = (
            "Each sub-question must be answerable using ONLY the available passages."
        )
    else:
        source_note = "available tables using data retrieved from a simple or advanced SQL query"
        answerable_note = (
            "Each sub-question must be answerable using ONLY the available tables. "
            "They may span multiple tables."
        )

    previous_block = ""
    if previous_subquestions:
        tried = "\n".join(f"- {q}" for q in previous_subquestions)
        previous_block = f"""
ALREADY TRIED SUB-QUESTIONS FOR THIS CLUSTER:
{tried}

Do not repeat or rephrase any of the above (same intent counts as a duplicate). Generate only new, distinct sub-questions.
"""

    empty_list_rule = (
        "- If none of the available tables or passages are relevant to the research question, return an empty list.\n"
        if allow_empty
        else ""
    )

    prompt = f"""Your task is to decompose the provided research question into smaller analytical sub-questions that can each be answered using the {source_note}. Make sure the sub-questions are phrased to be standalone –– the wording should contain all the relevant information needed to answer it, not being dependent on other sub-questions or the original research question or the list of available table/passage names.

RESEARCH QUESTION:
{user_query}

AVAILABLE CONTEXT:
{format_cluster_context(cluster)}
{previous_block}
Return ONLY valid JSON:
{{"subquestions": ["...", "...", ...]}}

Requirements:
- Provide up to {k} diverse sub-questions relevant to the research question and the available context.
{empty_list_rule}- Each sub-question must be one sentence.
- {answerable_note}
- Cover different analysis angles (counts, comparisons, trends, distributions).
- Do not repeat the full research question verbatim."""

    raw = chat(
        llm,
        prompt,
        temperature=temperature,
        feature="subquestion_generation",
        json_mode=True,
    )
    try:
        data = parse_json_dict_from_llm(raw)
    except ValueError as exc:
        log(f"Sub-question generation JSON parse failed: {exc}", silent=silent)
        return SubquestionGenerationResult([], parse_failed=True)
    items = data.get("subquestions") or data.get("questions") or []
    if not isinstance(items, list):
        log(
            "Sub-question generation returned non-list subquestions field.",
            silent=silent,
        )
        return SubquestionGenerationResult([], parse_failed=True)
    out = [str(q).strip() for q in items if str(q).strip()]
    return SubquestionGenerationResult(out[:k])


def filter_semantically_duplicate_subquestions(
    llm: LLMClient,
    answered: List[str],
    candidates: List[str],
    temperature: float,
) -> List[str]:
    """
    One LLM call: remove candidates equivalent to already-answered sub-questions.
    Returns the filtered candidate list.
    """
    if not answered or not candidates:
        return candidates

    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    answered_block = "\n".join(f"- {a}" for a in answered)

    prompt = f"""Your task is to remove questions from the "Candidate questions" list that are duplicate in meaning to one of the questions in the "Already answered" list.

Already answered:
{answered_block}

Candidate questions:
{numbered}

Return ONLY valid JSON:
{{"remove_indices": [1, 3]}}

List only the indices of candidate questions that need to be removed. Make sure to look for semantic equivalence to an already-answered question (same intent, not just similar wording). Return an empty list if none should be removed."""

    raw = chat(
        llm,
        prompt,
        temperature=temperature,
        feature="subquestion_dedup",
        json_mode=True,
    )
    try:
        data = parse_json_dict_from_llm(raw)
    except ValueError:
        return candidates
    remove = _parse_remove_indices(data.get("remove_indices"))
    if remove is None:
        return candidates

    kept = []
    for i, c in enumerate(candidates, start=1):
        if i not in remove:
            kept.append(c)
    return kept


def _fallback_subquestion(
    candidates: List[str],
    rng: random.Random,
    reason: str,
    *,
    silent: bool,
) -> str:
    chosen = rng.choice(candidates)
    log(
        f"Sub-question LLM selection failed ({reason}); falling back to random.",
        silent=silent,
    )
    return chosen


def _select_subquestion_with_llm(
    llm: LLMClient,
    *,
    user_query: str,
    cluster: dict,
    candidates: List[str],
    previous_subquestions: List[str],
    temperature: float,
    rng: random.Random,
    silent: bool = False,
) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(candidates, start=1))
    previous_block = (
        "\n".join(f"- {q}" for q in previous_subquestions)
        if previous_subquestions
        else "(none)"
    )

    prompt = f"""You are helping build a deep-research report that answers a research question.
Choose ONE candidate sub-question to investigate next.

RESEARCH QUESTION:
{user_query}

SELECTED CLUSTER CONTEXT:
{format_cluster_context(cluster)}

SUB-QUESTIONS ALREADY ASKED:
{previous_block}

CANDIDATE SUB-QUESTIONS:
{numbered}

{SUBQUESTION_SELECTION_RUBRIC.strip()}

Return ONLY valid JSON:
{{"selected_index": <1-based index>}}"""

    raw = chat(
        llm,
        prompt,
        temperature=temperature,
        feature="subquestion_selection",
        json_mode=True,
    )
    try:
        data = parse_json_dict_from_llm(raw)
        idx = int(data["selected_index"])
    except (ValueError, KeyError, TypeError) as exc:
        return _fallback_subquestion(
            candidates, rng, str(exc), silent=silent
        )

    if 1 <= idx <= len(candidates):
        log(
            f"LLM selected sub-question {idx} of {len(candidates)}.",
            silent=silent,
        )
        return candidates[idx - 1]
    return _fallback_subquestion(
        candidates, rng, f"index {idx} out of range", silent=silent
    )


def select_subquestion(
    candidates: List[str],
    method: str,
    rng: random.Random,
    *,
    llm: Optional[LLMClient] = None,
    user_query: str = "",
    cluster: Optional[dict] = None,
    previous_subquestions: Optional[List[str]] = None,
    temperature: float = 1.0,
    silent: bool = False,
) -> str:
    """Pick one sub-question from the candidate pool."""
    if not candidates:
        raise ValueError("No candidate sub-questions to select from.")

    if len(candidates) == 1:
        return candidates[0]

    if method == "random":
        return rng.choice(candidates)

    if method == "llm":
        if llm is None:
            raise ValueError("llm is required for subquestion_selection_method=llm")
        return _select_subquestion_with_llm(
            llm,
            user_query=user_query,
            cluster=cluster or {},
            candidates=candidates,
            previous_subquestions=previous_subquestions or [],
            temperature=temperature,
            rng=rng,
            silent=silent,
        )

    raise ValueError(f"Unknown subquestion_selection_method: {method}")
