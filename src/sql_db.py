"""SQLite helpers: load lake tables and run SQL for a cluster."""

import os
import re
import sqlite3
from typing import Any, Dict, List, Set, Tuple

from src.utils import load_json


def load_table_from_lake(tables_lake_dir: str, table_id: str) -> dict:
    path = os.path.join(tables_lake_dir, f"{table_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Table JSON not found: {path}")
    return load_json(path)


def load_cluster_table_metas(tables_lake_dir: str, table_ids: List[str]) -> Dict[str, dict]:
    metas = {}
    for tid in table_ids:
        metas[tid] = load_table_from_lake(tables_lake_dir, tid)
    return metas


def list_lake_table_ids(tables_lake_dir: str) -> List[str]:
    """Return table ids from schema_descriptions.json in the data lake."""
    schema_path = os.path.join(tables_lake_dir, "schema_descriptions.json")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"schema_descriptions.json not found: {schema_path}")
    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        raise ValueError(f"Expected object in {schema_path}")
    return list(schema.keys())


def materialize_lake_sqlite(tables_lake_dir: str, dest_path: str) -> str:
    """Build an on-disk SQLite DB containing every table in the data lake."""
    table_ids = list_lake_table_ids(tables_lake_dir)
    table_metas = load_cluster_table_metas(tables_lake_dir, table_ids)
    return materialize_cluster_sqlite_from_metas(table_metas, dest_path)


def materialize_cluster_sqlite(
    tables_lake_dir: str,
    table_ids: List[str],
    dest_path: str,
) -> str:
    """Build an on-disk SQLite DB containing only the given cluster tables."""
    table_metas = load_cluster_table_metas(tables_lake_dir, table_ids)
    return materialize_cluster_sqlite_from_metas(table_metas, dest_path)


def build_cluster_sqlite_artifact(
    table_metas: Dict[str, dict],
    dest_path: str,
) -> str:
    """Write cluster SQLite to disk and return the schema string for prompts."""
    conn, schema_string = build_cluster_sqlite(table_metas)
    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        if table_metas:
            dest = sqlite3.connect(dest_path)
            try:
                conn.backup(dest)
            finally:
                dest.close()
        else:
            with open(dest_path, "ab"):
                pass
    finally:
        conn.close()
    return schema_string


def materialize_cluster_sqlite_from_metas(
    table_metas: Dict[str, dict],
    dest_path: str,
) -> str:
    """Persist an in-memory cluster SQLite DB to disk."""
    build_cluster_sqlite_artifact(table_metas, dest_path)
    return os.path.abspath(dest_path)


def _sql_safe_identifier(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name or "").strip())
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"col_{s}"
    return s


def _sql_quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _unique_sql_column_name(base_safe: str, used_lower: Set[str]) -> str:
    """Assign a column name unique under SQLite's case-insensitive rules."""
    safe = base_safe
    if safe.lower() in used_lower:
        k = 2
        while f"{base_safe}_{k}".lower() in used_lower:
            k += 1
        safe = f"{base_safe}_{k}"
    used_lower.add(safe.lower())
    return safe


def _to_sqlite_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float)):
        return v     
    if isinstance(v, dict) and "value" in v:
        return v.get("value")
    # if isinstance(v, list) and v:
    #     first = v[0]
    #     if isinstance(first, (str, int, float)) or first is None:
    #         return first
    if isinstance(v, list):
        if not v:
            return None
        first = v[0]
        if isinstance(first, (str, int, float)) or first is None:
            return first
        return str(v)
    return str(v)

#format parser added
def _hybridqa_cell_value(cell: Any) -> Any:
    """Extract scalar from HybridQA cell: [value, links], {value: ...}, or scalar."""
    if isinstance(cell, list):
        return cell[0] if cell else None
    if isinstance(cell, dict) and "value" in cell:
        return cell.get("value")
    return cell


def build_cluster_sqlite(table_metas: Dict[str, dict]) -> Tuple[sqlite3.Connection, str]:
    """
    In-memory SQLite DB with one table per table_id (T1, T2, ...).
    Returns (connection, schema_string for LLM prompts).
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    schema_lines = []

    for table_id, meta in table_metas.items():
        cols = meta.get("columns") or []
        used_lower: Set[str] = set()
        col_map = {}
        col_defs = []
        schema_cols = []

        for c in cols:
            col_name = str(c)
            safe = _unique_sql_column_name(_sql_safe_identifier(col_name), used_lower)
            col_map[col_name] = safe
            col_defs.append(f"{_sql_quote_identifier(safe)} TEXT")
            schema_cols.append(f"{safe} (TEXT)")

        if not col_defs:
            col_defs = ["col TEXT"]
            schema_cols = ["col (TEXT)"]

        table_sql = _sql_quote_identifier(str(table_id))
        cursor.execute(f"DROP TABLE IF EXISTS {table_sql}")
        cursor.execute(f"CREATE TABLE {table_sql} ({', '.join(col_defs)})")
        schema_lines.append(f"Table: {table_id}, Columns: {', '.join(schema_cols)}")
        
        # data format cases modified and added
        rows_meta = meta.get("rows") or []
        logical_rows: List[Dict[str, Any]] = []

        if rows_meta and cols:
            sample = rows_meta[0]
            # Case 1: row-wise dicts
            if isinstance(sample, dict) and any(k in sample for k in cols):
                for r in rows_meta:
                    if isinstance(r, dict):
                        logical_rows.append({str(c): r.get(str(c)) for c in cols})
            # Case 2: flat list of {"value": ...} cells
            elif isinstance(sample, dict) and "value" in sample:
                step = len(cols)
                for i in range(0, len(rows_meta), step):
                    chunk = rows_meta[i : i + step]
                    if len(chunk) != step:
                        break
                    row_obj = {}
                    for col_name, cell in zip(cols, chunk):
                        row_obj[str(col_name)] = _hybridqa_cell_value(cell)
                    logical_rows.append(row_obj)
            # Case 3: HybridQA lake — each row is [[value, links], ...]
            elif isinstance(sample, list):
                for r in rows_meta:
                    if not isinstance(r, list):
                        continue
                    row_obj = {}
                    for col_name, cell in zip(cols, r):
                        row_obj[str(col_name)] = _hybridqa_cell_value(cell)
                    logical_rows.append(row_obj)
            

        if logical_rows:
            safe_cols = [col_map[str(c)] for c in cols]
            placeholders = ", ".join(["?"] * len(safe_cols))
            col_list_sql = ", ".join(_sql_quote_identifier(c) for c in safe_cols)
            insert_sql = (
                f"INSERT INTO {table_sql} ({col_list_sql}) VALUES ({placeholders})"
            )
            batch = [
                [_to_sqlite_scalar(r.get(str(c))) for c in cols] for r in logical_rows
            ]
            cursor.executemany(insert_sql, batch)

    conn.commit()
    return conn, "\n".join(schema_lines)


def fetch_sample_rows(conn: sqlite3.Connection, table_ids: List[str], limit: int = 3) -> str:
    """Example rows for SQL generation prompts."""
    lines = []
    cur = conn.cursor()
    for tid in table_ids:
        try:
            cur.execute(f'SELECT * FROM "{tid}" LIMIT {limit}')
            rows = cur.fetchall()
            colnames = [d[0] for d in cur.description] if cur.description else []
            lines.append(f"Table {tid} samples: {colnames} -> {rows[:limit]}")
        except Exception as e:
            lines.append(f"Table {tid}: (could not sample: {e})")
    return "\n".join(lines)


def execute_sql(conn: sqlite3.Connection, sql: str, row_limit: int = 50) -> Dict[str, Any]:
    """Run SQL and return status + preview rows."""
    sql = (sql or "").strip().rstrip(";")
    if not sql.upper().startswith("SELECT"):
        return {"ok": False, "error": "Only SELECT queries are allowed." + sql, "rows": []}

    # Safety: append LIMIT if missing
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {row_limit}"

    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        preview = [dict(zip(cols, row)) for row in rows[:row_limit]]
        return {"ok": True, "error": None, "row_count": len(rows), "rows": preview}
    except Exception as e:
        return {"ok": False, "error": str(e), "rows": []}


_SQL_TABLE_KEYWORDS = frozenset(
    {
        "select", "where", "on", "and", "or", "not", "null", "limit", "order",
        "group", "by", "having", "union", "as", "inner", "outer", "left",
        "right", "cross", "natural", "set", "all", "from", "join", "distinct",
    }
)


_TABLE_REF_RE = re.compile(
    r"""(?is)\b(?:from|join)\s+"""
    r'''(?:"([^"]+)"|'([^']+)'|`([^`]+)`|([a-zA-Z_]\w*))'''
)


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Return table ids referenced in FROM / JOIN clauses."""
    if not sql:
        return set()
    found: Set[str] = set()
    for match in _TABLE_REF_RE.finditer(sql):
        name = next(group for group in match.groups() if group is not None)
        if name.lower() not in _SQL_TABLE_KEYWORDS:
            found.add(name.upper())
    return found


def extract_sql_from_llm(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"(SELECT[\s\S]+)", text, re.IGNORECASE)
    return (m.group(1) if m else text).strip().rstrip(";")
