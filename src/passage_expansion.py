"""Expand cluster passages via ripgrep over passage descriptions."""

from __future__ import annotations

import bisect
import re
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Set

from src.llm import LLMClient, chat, parse_json_dict_from_llm
from src.sql_db import extract_tables_from_sql
from src.sql_qa import format_passages_for_prompt
from src.subquestions import format_cluster_context

MAX_ADDED_PASSAGES = 5
MAX_GREP_KEYWORDS = 5
EXPANSION_SUBQUESTIONS_K = 4

ExpansionMode = Literal["sql", "passages"]

_PASSAGE_KEY_LINE_RE = re.compile(r'^\s+"(P\d+)":\s*\{')


def format_sql_results_block(
    sql: str,
    execution_result: dict,
    *,
    max_rows: int = 100,
) -> str:
    if not execution_result.get("ok"):
        return f"SQL failed: {execution_result.get('error')}"
    rows = execution_result.get("rows") or []
    preview = str(rows[:max_rows])
    table_ids = sorted(extract_tables_from_sql(sql))
    tables_line = (
        f"Tables referenced: {', '.join(table_ids)}\n\n" if table_ids else ""
    )
    return (
        f"SQL Query:\n{sql}\n\n"
        f"{tables_line}"
        f"Results ({execution_result.get('row_count', 0)} rows):\n{preview}"
    )


def _parse_expansion_decision(data: dict) -> Dict[str, Any]:
    expand = bool(data.get("expand", False))
    keywords = [
        str(k).strip()
        for k in (data.get("grep_keywords") or [])
        if str(k).strip()
    ][:MAX_GREP_KEYWORDS]
    return {
        "expand": expand,
        "grep_keywords": keywords,
        "generate_new_subquestions": bool(data.get("generate_new_subquestions", False)),
    }


def _expansion_prompt_footer(*, expand_rule: str, keyword_source_rule: str) -> str:
    return f"""Return ONLY valid JSON:
{{
  "expand": true,
  "grep_keywords": ["keyword or short phrase", "another keyword"],
  "generate_new_subquestions": false
}}

Rules:
- {expand_rule}
- grep_keywords: 1-5 short search strings (OR semantics — any match counts).
  {keyword_source_rule}
- Set generate_new_subquestions to true only if the newly found passages would likely
  enable distinct follow-up sub-questions for this cluster.
- If expansion is not useful, return {{"expand": false, "grep_keywords": [], "generate_new_subquestions": false}}."""


def _sql_expansion_prompt(
    cluster: dict,
    sub_question: str,
    sql: str,
    execution_result: dict,
) -> str:
    footer = _expansion_prompt_footer(
        expand_rule=(
            "Set expand to true only if grep keywords derived from the SQL results "
            "could surface relevant passages not already in the cluster."
        ),
        keyword_source_rule=(
            "Use concrete terms from SQL result values (names, places, events), "
            "not generic words."
        ),
    )
    return f"""You are helping expand a research cluster with additional passages.

A sub-question will be answered using SQL over tables in the cluster. The SQL returned rows.
Decide whether searching the full passage description index with text keywords would help
answer the sub-question better or support follow-up analysis.

CURRENT CLUSTER:
{format_cluster_context(cluster)}

SUB-QUESTION:
{sub_question}

SQL EXECUTION:
{format_sql_results_block(sql, execution_result)}

{footer}"""


def _passages_expansion_prompt(cluster: dict, sub_question: str) -> str:
    passages = cluster.get("passages", [])
    footer = _expansion_prompt_footer(
        expand_rule=(
            "Set expand to true only if grep keywords could surface relevant passages "
            "not already in the cluster that would help answer the sub-question."
        ),
        keyword_source_rule=(
            "Use concrete terms from the sub-question and gaps in current passage "
            "coverage (entities, places, events mentioned in the question but weakly covered)."
        ),
    )
    return f"""You are helping expand a research cluster with additional passages.

A sub-question will be answered using only the passages currently in the cluster (no SQL).
Decide whether searching the full passage description index with text keywords would help
answer the sub-question better or support follow-up analysis.

CURRENT CLUSTER:
{format_cluster_context(cluster, include_passages=False)}

SUB-QUESTION:
{sub_question}

CURRENT PASSAGES (only evidence available):
{format_passages_for_prompt(passages)}

{footer}"""


def decide_passage_expansion(
    llm: LLMClient,
    cluster: dict,
    sub_question: str,
    temperature: float,
    *,
    mode: ExpansionMode,
    sql: Optional[str] = None,
    execution_result: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Ask the LLM whether to grep passage_descriptions.json for extra passages.

    Returns parsed JSON with keys:
      expand (bool), grep_keywords (list[str]), generate_new_subquestions (bool)
    """
    if mode == "sql":
        if not sql or execution_result is None:
            raise ValueError("sql and execution_result are required for mode='sql'")
        prompt = _sql_expansion_prompt(cluster, sub_question, sql, execution_result)
        feature = "passage_expansion_decide_sql"
    else:
        prompt = _passages_expansion_prompt(cluster, sub_question)
        feature = "passage_expansion_decide_passages"

    raw = chat(
        llm,
        prompt,
        temperature=temperature,
        feature=feature,
        json_mode=True,
    )
    try:
        data = parse_json_dict_from_llm(raw)
    except ValueError:
        return _parse_expansion_decision({})
    return _parse_expansion_decision(data)


@lru_cache(maxsize=4)
def _passage_key_line_index(passage_descriptions_path: str) -> tuple[tuple[int, str], ...]:
    """Map each top-level passage key line in passage_descriptions.json to its id."""
    entries: List[tuple[int, str]] = []
    with open(passage_descriptions_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            match = _PASSAGE_KEY_LINE_RE.match(line)
            if match:
                entries.append((line_no, match.group(1)))
    return tuple(entries)


def _passage_id_for_line(
    line_no: int,
    key_lines: tuple[tuple[int, str], ...],
) -> Optional[str]:
    if not key_lines:
        return None
    line_numbers = [item[0] for item in key_lines]
    idx = bisect.bisect_right(line_numbers, line_no) - 1
    if idx < 0:
        return None
    return key_lines[idx][1]


def _rg_or_pattern(keywords: List[str]) -> str:
    return "|".join(re.escape(keyword) for keyword in keywords if keyword)


def grep_passage_ids(
    passage_descriptions_path: str,
    keywords: List[str],
    *,
    exclude_ids: Set[str],
    max_results: int = MAX_ADDED_PASSAGES,
    rank_by: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    Return up to max_results passage ids matching any keyword (OR semantics).

    Uses ripgrep over passage_descriptions.json, then maps match lines back to
    passage ids. Results are sorted by embedding rank to the original query
    (lower rank first). Passages without a rank sort after ranked ones.
    """
    if not keywords:
        return []

    pattern = _rg_or_pattern(keywords)
    if not pattern:
        return []

    proc = subprocess.run(
        [
            "rg",
            "-i",
            "-n",
            "--no-heading",
            "--no-filename",
            "--color",
            "never",
            pattern,
            passage_descriptions_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"ripgrep failed (exit {proc.returncode})"
            + (f": {stderr}" if stderr else "")
        )

    key_lines = _passage_key_line_index(passage_descriptions_path)
    matches: Set[str] = set()
    for raw_line in (proc.stdout or "").splitlines():
        if not raw_line.strip():
            continue
        try:
            line_no = int(raw_line.split(":", 1)[0])
        except (ValueError, IndexError):
            continue
        passage_id = _passage_id_for_line(line_no, key_lines)
        if passage_id and passage_id not in exclude_ids:
            matches.add(passage_id)

    def sort_key(pid: str) -> tuple:
        rank = rank_by.get(pid) if rank_by else None
        if rank is None:
            return (1, _passage_id_number(pid))
        return (0, rank)

    ordered = sorted(matches, key=sort_key)
    return ordered[:max_results]


def _passage_id_number(passage_id: str) -> int:
    digits = "".join(ch for ch in passage_id if ch.isdigit())
    return int(digits) if digits else 0


def passages_from_ids(
    passage_descriptions: Dict[str, dict],
    passage_ids: List[str],
) -> List[dict]:
    """Build cluster-shaped passage dicts from passage_descriptions records."""
    passages: List[dict] = []
    for pid in passage_ids:
        record = passage_descriptions.get(pid)
        if not record:
            continue
        passages.append(
            {
                "passage_id": pid,
                "uid": record.get("uid", ""),
                "title": record.get("title", ""),
                "text": record.get("text", ""),
            }
        )
    return passages
