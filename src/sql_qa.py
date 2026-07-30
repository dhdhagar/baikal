"""Generate SQL and natural-language answers from query results."""

from typing import List, Optional

from src.llm import LLMClient, chat, parse_json_dict_from_llm
from src.passage_embeddings import build_passage_text
from src.sql_db import extract_sql_from_llm, extract_tables_from_sql


def format_passages_for_prompt(passages: List[dict], max_chars: Optional[int] = 12000) -> str:
    if not passages:
        return "(none)"
    blocks = []
    total = 0
    for p in passages:
        block = (
            f"[{p.get('passage_id', 'passage')}] "
            f"{build_passage_text(p)}"
        )
        if max_chars is not None and total + len(block) > max_chars:
            blocks.append("...(additional passages truncated)")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def decide_sql_needed(
    llm: LLMClient,
    sub_question: str,
    passages: List[dict],
    schema_string: str,
    temperature: float,
) -> bool:
    """Return True when SQL over tables is needed to answer the question."""
    if not schema_string.strip():
        return False

    prompt = f"""You are deciding how to answer a question using available tables and passages.

Question:
{sub_question}

Available tables (SQL schema):
{schema_string or "(none)"}

Available passages:
{format_passages_for_prompt(passages)}

Return ONLY valid JSON:
{{"needs_sql": true}}

Set needs_sql to true only if answering the question requires querying the tables via SQL.
Set needs_sql to false if the passages alone are sufficient."""

    raw = chat(
        llm,
        prompt,
        temperature=temperature,
        max_tokens=None,
        feature="sql_decide",
        json_mode=True,
    )
    try:
        data = parse_json_dict_from_llm(raw)
    except ValueError:
        return True
    return bool(data.get("needs_sql", True))


def generate_sql(
    llm: LLMClient,
    sub_question: str,
    schema_string: str,
    samples_string: str,
    temperature: float,
    max_tokens: int = 8192,
) -> str:
    system_prompt = """You are a SQLite expert."""
    prompt = f"""You are given a question and a schema. Your task is to write ONE SELECT query that answers the provided question using ONLY the tables and columns from the provided schema.

Schema:
{schema_string}

Example rows:
{samples_string}

Question: {sub_question}

Rules:
- Use only tables/columns from the provided schema.
- Output ONLY the SQL statement (no markdown, no other text).
- You may use aggregates (COUNT, GROUP BY, MIN, MAX, SUM, AVG, etc.), for instance, when the question asks for patterns, distributions, counts, rankings, comparisons, or ranges.
- When appropriate, you may use advanced SQL features (joins, subqueries, window functions, etc.).
- Make sure you sort the results by the most relevant column(s) when appropriate so that the results are easy to understand.
- Never invent tables or columns that are not present in the provided schema."""

    raw = chat(
        llm,
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        feature="sql_generate",
    )
    return extract_sql_from_llm(raw)


def refine_sql_with_error(
    llm: LLMClient,
    sub_question: str,
    schema_string: str,
    samples_string: str,
    previous_sql: str,
    error_message: Optional[str],
    empty_result: bool,
    temperature: float,
    max_tokens: int = 8192,
    attempts_log: Optional[List[dict]] = None,
) -> str:
    """
    SQL refinement step (Stage-3-style): debug observed errors or empty results
    and return a corrected query.
    """
    attempts_text = ""
    if attempts_log:
        last = attempts_log[-3:]
        attempts_text = "\n".join(
            f"- attempt {a.get('attempt')}: ok={a.get('ok')} row_count={a.get('row_count')} error={a.get('error')}"
            for a in last
        )

    system_prompt = "You are a SQLite debugging assistant."
    prompt = f"""You previously wrote a SQL query for a question, but it failed or returned 0 rows. Your task is to refine the query to fix the issue.

Schema:
{schema_string}

Example rows:
{samples_string}

Question:
{sub_question}

Previous SQL:
{previous_sql}

Observed issue:
- error_message: {error_message or "(none)"}
- empty_result (0 rows): {empty_result}

Recent attempts:
{attempts_text or "(none)"}

Rules:
- Use only tables/columns from the provided schema.
- Output ONLY the SQL statement (no markdown, no other text).
- You may use aggregates (COUNT, GROUP BY, MIN, MAX, SUM, AVG, etc.), for instance, when the question asks for patterns, distributions, counts, rankings, comparisons, or ranges.
- When appropriate, you may use advanced SQL features (joins, subqueries, window functions, etc.).
- Make sure you sort the results by the most relevant column(s) when appropriate so that the results are easy to understand.
- Never invent tables or columns that are not present in the provided schema.
- If the previous SQL returned an empty result, simplify the query.
- Prefer single-table unless there is a clear join key shared by tables.
"""
    raw = chat(
        llm,
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        feature="sql_refine",
    )
    return extract_sql_from_llm(raw)


def _citation_rules(*, has_passages: bool, has_sql: bool) -> str:
    rules = [
        "- Ground every factual claim in the provided evidence only.",
        "- Do not invent facts beyond the evidence.",
    ]
    if has_passages:
        rules.append(
            "- When a claim comes from a passage, cite its ID in brackets "
            "(e.g., [P42])."
        )
    if has_sql:
        rules.append(
            "- When a claim comes from SQL results, cite the table ID(s) "
            "that supplied the data (e.g., [T7280])."
        )
    return "\n".join(rules)


def answer_from_context(
    llm: LLMClient,
    sub_question: str,
    passages: List[dict],
    sql: Optional[str] = None,
    execution_result: Optional[dict] = None,
    temperature: float = 1.0,
    max_rows: Optional[int] = 100,
    max_tokens: int = 4096,
) -> str:
    """Answer using passages and, when provided, SQL results."""
    passage_block = format_passages_for_prompt(passages)
    has_passages = bool(passages)
    has_sql = bool(sql and execution_result is not None)

    sql_block = ""
    if has_sql:
        if not execution_result.get("ok"):
            sql_block = f"SQL failed: {execution_result.get('error')}"
        else:
            rows = execution_result.get("rows") or []
            preview = str(rows[:max_rows])
            table_ids = sorted(extract_tables_from_sql(sql))
            tables_line = (
                f"Tables referenced: {', '.join(table_ids)}\n\n"
                if table_ids
                else ""
            )
            sql_block = (
                f"SQL Query:\n{sql}\n\n"
                f"{tables_line}"
                f"Results ({execution_result.get('row_count', 0)} rows):\n{preview}"
            )

    prompt = f"""Answer the question using ONLY the provided passages and SQL results (if any). Be concise (2-5 sentences). Mention key numbers. If evidence is insufficient, say so.

Question: {sub_question}

Passages:
{passage_block}
"""
    if sql_block:
        prompt += f"\nSQL results:\n{sql_block}\n"

    prompt += f"""
Citation rules:
{_citation_rules(has_passages=has_passages, has_sql=has_sql)}
"""

    return chat(
        llm,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        feature="answer_from_context",
    )


def answer_from_sql_results(
    llm: LLMClient,
    sub_question: str,
    sql: str,
    execution_result: dict,
    temperature: float,
    max_rows: Optional[int] = 100,
    max_tokens: int = 4096,
) -> str:
    """Turn SQL output into a short grounded answer."""
    if not execution_result.get("ok"):
        return f"Could not answer: {execution_result.get('error')}"

    rows = execution_result.get("rows") or []
    preview = str(rows[:max_rows])
    table_ids = sorted(extract_tables_from_sql(sql))
    tables_line = (
        f"Tables referenced: {', '.join(table_ids)}\n\n" if table_ids else ""
    )

    prompt = f"""Answer the question using ONLY the SQL results below. Be concise (2-5 sentences). Mention key numbers. If results are empty, say so.

Question: {sub_question}

SQL Query:
{sql}

{tables_line}Results ({execution_result.get('row_count', 0)} rows{", showing up to " + str(max_rows) if max_rows else ""}):
{preview}

Citation rules:
{_citation_rules(has_passages=False, has_sql=True)}
"""
    return chat(
        llm,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        feature="answer_from_sql",
    )
