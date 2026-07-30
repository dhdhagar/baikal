"""OpenCode baseline: minimal agent harness with budgeted lake access."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from src.cli_ui import ConsoleUI, get_ui, init_ui
from src.embedding_client import get_embedding_client
from src.inference_clusters import build_topk_inference_cluster
from src.embeddings import (
    filter_clusters_by_tables_passages,
    get_passage_similarity_ranks,
    get_table_similarity_ranks,
    top_k_ids_from_ranks,
    top_k_table_ids_from_ranks,
)
from src.llm import get_llm_client
from src.metrics.common import (
    assign_finding_indices,
    count_lake_tables,
    extract_gt_passage_ids,
    extract_gt_table_ids,
    is_report_eligible_iteration,
    load_uid_to_passage_id_mapping,
    load_uid_to_table_id_mapping,
    sync_finding_indices_to_query_dir,
)
from src.metrics.progress import make_metrics_reporter
from src.metrics.tracker import MetricsTracker
from src.metrics.research_quality import build_judge_clients
from src.opencode_lake_tool import (
    _execution_from_attempt,
    init_state,
    load_config,
    load_state,
    save_state,
    write_config,
)
from src.queries import resolve_queries
from src.report import generate_final_report
from src.result_json import (
    INITIAL_CLUSTERS_FILENAME,
    build_result_json,
    result_usage,
    save_query_result,
)
from src.sql_db import materialize_lake_sqlite
from src.opencode_usage import (
    build_opencode_run_usage,
    combine_query_usage,
    fetch_all_opencode_session_usage,
    format_nested_usage_line,
    load_latest_opencode_session_id,
    parse_session_id_from_json_stdout,
)
from src.tracking import UsageSummary, get_tracker, reset_tracker
from src.utils import (
    ensure_dir,
    load_json,
    log,
    passage_descriptions_filename,
    query_passage_descriptions_path,
    resolve_passage_descriptions_source,
    save_json,
    set_seed,
)

DEFAULT_OPENCODE_TIMEOUT_SEC = 7200
DEFAULT_MAX_OPENCODE_CONTINUATIONS = 24
DEFAULT_LAKE_UI_POLL_SEC = 0.5
INITIAL_RETRIEVAL_FILENAME = "initial_retrieval.json"
OPENCODE_DATA_DIRNAME = ".opencode_data"
OPENCODE_ROUND_SEPARATOR = "\n\n--- opencode continuation ---\n\n"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAKE_TOOL_STATE_FILENAME = "lake_tool_state.json"


@dataclass(frozen=True)
class PartialWorkspaceStatus:
    resume_agent: bool = False
    finalize_only: bool = False
    n_findings: int = 0


def _coerce_subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def partial_workspace_status(query_dir: str, budget: int) -> PartialWorkspaceStatus:
    """Detect whether a query workspace can resume the agent or only finalize."""
    if os.path.isfile(os.path.join(query_dir, "result.json")):
        return PartialWorkspaceStatus()

    state_path = os.path.join(query_dir, LAKE_TOOL_STATE_FILENAME)
    if not os.path.isfile(state_path):
        return PartialWorkspaceStatus()

    try:
        state = load_json(state_path)
    except (OSError, ValueError):
        return PartialWorkspaceStatus()

    n_findings = len(state.get("findings") or [])
    if n_findings <= 0:
        return PartialWorkspaceStatus()
    if n_findings >= budget:
        return PartialWorkspaceStatus(
            finalize_only=True,
            n_findings=n_findings,
        )
    return PartialWorkspaceStatus(
        resume_agent=True,
        n_findings=n_findings,
    )


def _load_resume_workspace_context(query_dir: str) -> Dict[str, Any]:
    """Reload top-k and cluster metadata from saved initial artifacts."""
    initial_retrieval_path = os.path.join(query_dir, INITIAL_RETRIEVAL_FILENAME)
    initial_clusters_path = os.path.join(query_dir, INITIAL_CLUSTERS_FILENAME)

    topk_table_ids: List[str] = []
    topk_passage_ids: List[str] = []
    if os.path.isfile(initial_retrieval_path):
        payload = load_json(initial_retrieval_path)
        topk_table_ids = [
            str(item["id"])
            for item in (payload.get("tables") or [])
            if item.get("id") is not None
        ]
        topk_passage_ids = [
            str(item["id"])
            for item in (payload.get("passages") or [])
            if item.get("id") is not None
        ]
    else:
        initial_retrieval_path = None

    retained_clusters = 0
    total_inference_clusters = 0
    metrics_inference_clusters: List[dict] = []
    if os.path.isfile(initial_clusters_path):
        clusters_payload = load_json(initial_clusters_path)
        retained_clusters = int(clusters_payload.get("n_retained_clusters") or 0)
        total_inference_clusters = int(
            clusters_payload.get("n_total_inference_clusters") or 0
        )
        metrics_inference_clusters = list(clusters_payload.get("clusters") or [])
    else:
        initial_clusters_path = None

    return {
        "topk_table_ids": topk_table_ids,
        "topk_passage_ids": topk_passage_ids,
        "initial_retrieval_path": initial_retrieval_path,
        "initial_clusters_path": initial_clusters_path,
        "retained_clusters": retained_clusters,
        "total_inference_clusters": total_inference_clusters,
        "metrics_inference_clusters": metrics_inference_clusters,
    }


def _shared_opencode_data_dir() -> str:
    """Canonical OpenCode credential store (not the per-workspace XDG_DATA_HOME)."""
    override = os.environ.get("OPENCODE_SHARED_DATA_HOME")
    if override:
        return os.path.join(os.path.expanduser(override), "opencode")
    return os.path.join(os.path.expanduser("~/.local/share"), "opencode")


def _ensure_opencode_credentials(data_home: str) -> None:
    """Link shared OpenCode credentials into an isolated per-workspace data home."""
    target_dir = ensure_dir(os.path.join(data_home, "opencode"))
    shared_dir = _shared_opencode_data_dir()
    for name in ("auth.json", "account.json"):
        src = os.path.join(shared_dir, name)
        dest = os.path.join(target_dir, name)
        if not os.path.isfile(src) or os.path.lexists(dest):
            continue
        try:
            os.symlink(os.path.abspath(src), dest)
        except OSError:
            shutil.copy2(src, dest)


def _opencode_data_home(cwd: str) -> str:
    """Per-workspace OpenCode state so parallel workers don't share opencode.db."""
    data_home = ensure_dir(os.path.join(cwd, OPENCODE_DATA_DIRNAME))
    _ensure_opencode_credentials(data_home)
    return data_home


def _opencode_subprocess_env(cwd: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    env["XDG_DATA_HOME"] = _opencode_data_home(cwd)
    return env


def _opencode_skip_retrieval(args: Namespace) -> bool:
    return getattr(args, "opencode_skip_retrieval", True)


def _opencode_skip_clustering(args: Namespace) -> bool:
    return getattr(args, "opencode_skip_clustering", True)


def load_opencode_inference_clusters(
    args: Namespace,
    *,
    inference_clusters_path: Optional[str] = None,
) -> List[dict]:
    """
    Load global inference clusters for OpenCode metrics and cluster artifacts.

    When cluster artifacts are enabled, load BERTopic clusters from disk only if
    use_clustering is true (per-query top-k clusters are built later in-query).
    When cluster artifacts are skipped, still load a cached file when present so
    retrieval metrics can use corpus cluster definitions.
    """
    path = inference_clusters_path or getattr(args, "inference_clusters_path", None)
    if not path or not os.path.isfile(path):
        return []
    if _opencode_skip_clustering(args):
        return load_json(path)
    if args.use_clustering:
        return load_json(path)
    return []


def setup_opencode_inference_clusters(args: Namespace) -> List[dict]:
    """Ensure or load inference clusters before an OpenCode run."""
    if not _opencode_skip_clustering(args):
        from src.clustering import ensure_inference_clusters

        return ensure_inference_clusters(args)
    return load_opencode_inference_clusters(args)


def _log_opencode_cluster_setup(
    args: Namespace,
    inference_clusters: List[dict],
    *,
    silent: bool = False,
    use_print: bool = False,
) -> None:
    if _opencode_skip_clustering(args):
        return
    if args.use_clustering:
        message = f"Inference clusters: {len(inference_clusters)}"
    else:
        message = "Clustering disabled; one top-k cluster per query."
    if use_print:
        print(message)
    else:
        log(message, silent=silent)


def _lake_tool_cmd(workdir: str, *args: str) -> str:
    return " ".join(
        [
            f"PYTHONPATH={PROJECT_ROOT}",
            sys.executable,
            "-m",
            "src.opencode_lake_tool",
            "--workdir",
            workdir,
            *args,
        ]
    )


def build_opencode_prompt(
    *,
    query_id: str,
    user_query: str,
    schema_descriptions_path: str,
    sqlite_db_path: str,
    query_dir: str,
    budget: int,
    max_sql_attempts: int,
    use_passages: bool,
    passage_descriptions_path: Optional[str],
    initial_retrieval_path: Optional[str] = None,
    initial_clusters_path: Optional[str] = None,
) -> str:
    passage_block = ""
    if use_passages and passage_descriptions_path:
        passage_block = f"""
Passage descriptions: {passage_descriptions_path}
Load a passage: {_lake_tool_cmd(query_dir, "passage", "P42")}
"""

    retrieval_block = ""
    if initial_retrieval_path:
        if use_passages:
            retrieval_detail = (
                "This JSON lists embedding-ranked table/passage ids and titles as starting points.\n"
                "Full schema and passage metadata remain in the files above."
            )
        else:
            retrieval_detail = (
                "This JSON lists embedding-ranked table ids and titles as starting points.\n"
                "Full schema metadata remains in the files above."
            )
        retrieval_block = f"""
Initial retrieval candidates: {initial_retrieval_path}
{retrieval_detail}
"""

    clusters_block = ""
    if initial_clusters_path:
        if use_passages:
            clusters_detail = (
                "Topic-grouped table/passage clusters overlapping the retrieval "
                "candidates above.\n"
                "Each cluster has cluster_id, description, tables, and passages."
            )
        else:
            clusters_detail = (
                "Topic-grouped table clusters overlapping the retrieval candidates "
                "above.\n"
                "Each cluster has cluster_id, description, and tables."
            )
        clusters_block = f"""
Initial inference clusters: {initial_clusters_path}
{clusters_detail}
"""

    commit_sql_example = _lake_tool_cmd(
        query_dir,
        "commit",
        "--sub-question",
        '"What is the stadium capacity?"',
        "--answer",
        '"The capacity is 50,000 [T809]."',
    )
    citation_rules = f"""Citation rules (required in every --answer):
- Ground every factual claim in lake evidence only.
- When a claim comes from SQL results, cite the table ID(s) in brackets (e.g., [T809]).
- SQL-backed commit example:
  {commit_sql_example}"""
    if use_passages and passage_descriptions_path:
        commit_passage_example = _lake_tool_cmd(
            query_dir,
            "commit",
            "--sub-question",
            '"What does the passage say about density?"',
            "--answer",
            '"Urban density is higher in the core [P42]."',
            "--no-sql",
        )
        citation_rules += f"""
- When a claim comes from a passage, cite its ID in brackets (e.g., [P42]).
- Passage-only commit example:
  {commit_passage_example}"""

    finding_quality_rules = """Finding quality rules (required):
- Do not commit findings that only report table row counts (e.g. SELECT COUNT(*) FROM T…).
- Do not spend findings on schema discovery (sqlite_master, listing tables, column introspection).
- Each finding must contain substantive analysis that advances the research question."""
    if initial_retrieval_path:
        finding_quality_rules += (
            "\n- Start by looking at the initial retrieval candidates above before exploring the full lake."
        )

    return f"""You are a research agent answering a question using only the provided data lake.

Research question:
{user_query}

Query id: {query_id}

Schema descriptions: {schema_descriptions_path}
SQLite database: {sqlite_db_path}
Working directory: {query_dir}
{retrieval_block}{clusters_block}
Budget: record at most {budget} findings. Each finding is one evidence-gathering step.
Per finding you may run up to {max_sql_attempts} SQL attempts before committing it.

Run SQL (table names may be unquoted or double-quoted, e.g. T809 or "T809"):
{_lake_tool_cmd(query_dir, "sql", '"SELECT * FROM T809 LIMIT 10"')}

Record a finding (required after each step):
{_lake_tool_cmd(query_dir, "commit", "--sub-question", '"..."', "--answer", '"..."')}
{_lake_tool_cmd(query_dir, "commit", "--sub-question", '"..."', "--answer", '"..."', "--no-sql")}

{citation_rules}

{finding_quality_rules}

Check budget status:
{_lake_tool_cmd(query_dir, "status")}
{passage_block}
Use only lake evidence. Do not use outside knowledge. Do not invent facts beyond the evidence.
Do not modify the lake in any way.

Autonomous execution (required):
- You are running fully unattended; no human is available to answer questions.
- Never ask questions, present option menus, or end a turn waiting for user input.
- If multiple next steps are possible, choose the one most likely to gather evidence and execute it immediately.
- Keep running sql/commit cycles until `status` reports the budget is exhausted ({budget} findings).
"""


def _compute_relevance(
    args: Namespace,
    user_query: str,
    embedder,
) -> tuple[List[str], List[str]]:
    table_rank_by_id = get_table_similarity_ranks(
        user_query,
        args.table_embeddings_path,
        args.embedding_model,
        embedding_provider=args.embedding_provider,
        embedder=embedder,
    )
    topk_table_ids = top_k_table_ids_from_ranks(
        table_rank_by_id,
        args.k_relevant_tables,
    )
    topk_passage_ids: List[str] = []
    if args.use_passages:
        passage_rank_by_id = get_passage_similarity_ranks(
            user_query,
            args.passage_embeddings_path,
            args.embedding_model,
            embedding_provider=args.embedding_provider,
            embedder=embedder,
        )
        topk_passage_ids = top_k_ids_from_ranks(
            passage_rank_by_id,
            args.k_relevant_passages,
        )
    return topk_table_ids, topk_passage_ids


def _build_initial_retrieval_payload(
    topk_table_ids: List[str],
    topk_passage_ids: List[str],
    *,
    schema_descriptions_path: str,
    passage_descriptions_path: Optional[str],
) -> Dict[str, Any]:
    schema = load_json(schema_descriptions_path)
    tables = [
        {
            "id": table_id,
            "name": str((schema.get(table_id) or {}).get("title") or ""),
        }
        for table_id in topk_table_ids
    ]
    passages: List[Dict[str, str]] = []
    if topk_passage_ids and passage_descriptions_path:
        passage_meta = load_json(passage_descriptions_path)
        passages = [
            {
                "id": passage_id,
                "name": str((passage_meta.get(passage_id) or {}).get("title") or ""),
            }
            for passage_id in topk_passage_ids
        ]
    return {"tables": tables, "passages": passages}


def write_initial_retrieval_artifact(
    query_dir: str,
    payload: Dict[str, Any],
) -> str:
    path = os.path.join(query_dir, INITIAL_RETRIEVAL_FILENAME)
    save_json(path, payload)
    return path


def _build_initial_clusters_payload(
    inference_clusters: List[dict],
    topk_table_ids: List[str],
    topk_passage_ids: List[str],
    *,
    args: Namespace,
    passage_descriptions_path: Optional[str],
) -> Dict[str, Any]:
    if not args.use_clustering:
        query_clusters = build_topk_inference_cluster(
            topk_table_ids,
            topk_passage_ids,
            tables_lake_dir=args.tables_lake_dir,
            passage_descriptions_path=passage_descriptions_path,
        )
    else:
        query_clusters = inference_clusters
    retained = filter_clusters_by_tables_passages(
        query_clusters,
        topk_table_ids,
        topk_passage_ids=topk_passage_ids if args.use_passages else None,
    )
    return {
        "n_total_inference_clusters": len(query_clusters),
        "n_retained_clusters": len(retained),
        "clusters": retained,
    }


def write_initial_clusters_artifact(
    query_dir: str,
    payload: Dict[str, Any],
) -> str:
    path = os.path.join(query_dir, INITIAL_CLUSTERS_FILENAME)
    save_json(path, payload)
    return path


def _materialize_passage_descriptions(
    query_dir: str,
    source_path: str,
    passage_type: str,
) -> str:
    """Link or copy passage descriptions into the query workspace for the agent."""
    dest = query_passage_descriptions_path(query_dir, passage_type)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"{passage_descriptions_filename(passage_type)} not found at {source_path}")
    if os.path.lexists(dest):
        return dest
    abs_source = os.path.abspath(source_path)
    try:
        os.symlink(abs_source, dest)
    except OSError:
        shutil.copy2(abs_source, dest)
    return dest


def setup_lake_workspace(
    args: Namespace,
    workdir: str,
    sqlite_db_path: str,
    *,
    budget: Optional[int] = None,
    cluster_passage_ids: Optional[List[str]] = None,
    allow_passage_grep: bool = False,
    passage_rank_by_id: Optional[Dict[str, int]] = None,
    reset_state: bool = True,
) -> None:
    passage_descriptions_path = None
    if args.use_passages:
        passage_descriptions_path = _materialize_passage_descriptions(
            workdir,
            resolve_passage_descriptions_source(args),
            args.passage_type,
        )
    write_config(
        workdir,
        {
            "sqlite_path": sqlite_db_path,
            "passage_descriptions_path": passage_descriptions_path,
            "budget": budget if budget is not None else args.budget,
            "max_sql_attempts": args.max_sql_attempts,
            "allow_passage_grep": allow_passage_grep,
            "cluster_passage_ids": cluster_passage_ids or [],
            "passage_rank_by_id": passage_rank_by_id or {},
        },
    )
    if reset_state:
        save_state(workdir, init_state())


def _setup_query_workspace(
    args: Namespace,
    query_dir: str,
    sqlite_db_path: str,
    *,
    reset_state: bool = True,
) -> None:
    setup_lake_workspace(
        args,
        query_dir,
        sqlite_db_path,
        reset_state=reset_state,
    )


def findings_to_iterations(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "step": int(item["step"]),
            "sub_question": item.get("sub_question") or "",
            "answer": item.get("answer") or "",
            "needs_sql": bool(item.get("needs_sql")),
            "sql": item.get("sql"),
            "execution": item.get("execution"),
            "failed_sql_attempts": item.get("failed_sql_attempts"),
        }
        for item in findings
    ]


def write_iteration_artifacts(
    query_dir: str,
    iterations: List[Dict[str, Any]],
    budget: int,
) -> List[Dict[str, Any]]:
    """Write iteration_XXX.json for every budget step; return recorded findings only."""
    by_step = {int(it["step"]): it for it in iterations}
    recorded: List[Dict[str, Any]] = []

    for step in range(1, budget + 1):
        if step in by_step:
            payload = dict(by_step[step])
            recorded.append(payload)
            save_json(os.path.join(query_dir, f"iteration_{step:03d}.json"), payload)
        else:
            skipped = {"step": step, "skipped": True}
            save_json(os.path.join(query_dir, f"iteration_{step:03d}.json"), skipped)

    return recorded


def _build_metrics_tracker(
    args: Namespace,
    *,
    query_record: Dict[str, Any],
    user_query: str,
    query_dir: str,
    topk_table_ids: List[str],
    topk_passage_ids: List[str],
    inference_clusters: List[dict],
    rich: bool,
    initial_candidate_clusters: int = 0,
) -> Optional[MetricsTracker]:
    if not args.compute_metrics:
        return None

    ground_truth = query_record.get("ground_truth")
    uid_to_table_id = load_uid_to_table_id_mapping(args.uid_to_table_id_path)
    gt_tables = extract_gt_table_ids(ground_truth, uid_to_table_id=uid_to_table_id)
    gt_passages = None
    if args.use_passages:
        uid_to_passage_id = load_uid_to_passage_id_mapping(args.uid_to_passage_id_path)
        gt_passages = extract_gt_passage_ids(
            ground_truth,
            uid_to_passage_id=uid_to_passage_id,
            passage_type=args.passage_type,
        )

    from src.utils import load_passage_descriptions_for_metrics

    return MetricsTracker(
        user_query=user_query,
        budget=args.budget,
        gt_tables=gt_tables,
        total_lake_tables=count_lake_tables(args.tables_lake_dir),
        topk_table_ids=topk_table_ids,
        inference_clusters=inference_clusters,
        initial_candidate_clusters=initial_candidate_clusters,
        k_relevant_tables=args.k_relevant_tables,
        embedder=get_embedding_client(
            args.embedding_provider,
            args.embedding_model,
            gpu=args.gpu,
        )
        if args.compute_embed_diversity
        else None,
        compute_embed_diversity=bool(args.compute_embed_diversity),
        gt_passages=gt_passages,
        topk_passage_ids=topk_passage_ids,
        k_relevant_passages=args.k_relevant_passages,
        use_passages=args.use_passages,
        judge_llms=(
            build_judge_clients(args.llm_provider, args.judge_models)
            if not args.no_llm_judge
            else []
        ),
        judge_temperature=1.0,
        research_quality_enabled=not args.no_llm_judge,
        reporter=make_metrics_reporter(
            args.budget,
            mode="auto",
            silent=args.silent,
            rich_cli=rich,
        ),
        passage_descriptions=load_passage_descriptions_for_metrics(
            args,
            query_dir=query_dir,
        ),
    )


def _record_metrics(
    metrics_tracker: Optional[MetricsTracker],
    iterations: List[Dict[str, Any]],
    budget: int,
) -> None:
    if not metrics_tracker:
        return

    by_step = {int(it["step"]): it for it in iterations if it.get("step")}
    for step in range(1, budget + 1):
        iteration = by_step.get(step)
        if iteration:
            metrics_tracker.record_iteration(iteration, step, 0)
        else:
            metrics_tracker.record_empty_step(step, 0)


def _findings_count(workdir: str) -> int:
    state = load_state(workdir)
    return len(state.get("findings") or [])


@dataclass
class _LakeUISeen:
    seen_steps: set[int] = field(default_factory=set)
    n_attempts: int = 0

    @property
    def n_findings(self) -> int:
        return len(self.seen_steps)


def _apply_lake_state_to_ui(
    workdir: str,
    ui: ConsoleUI,
    *,
    budget: int,
    seen: _LakeUISeen,
) -> None:
    """Mirror lake_tool_state.json into the Rich iteration panel."""
    state = load_state(workdir)
    findings = list(state.get("findings") or [])
    work_step = int(state.get("step") or 1)

    new_findings = [
        finding
        for finding in findings
        if int(finding.get("step") or 0) not in seen.seen_steps
    ]
    if new_findings:
        new_findings.sort(key=lambda finding: int(finding.get("step") or 0))
        for finding in new_findings:
            step = int(finding.get("step") or 0)
            ui.set_budget_step(step)
            ui.set_sub_question(finding.get("sub_question") or "")
            ui.set_sql_result(
                finding.get("sql") or "",
                finding.get("execution"),
            )
            ui.show_sql_answer(finding.get("answer") or "", pause_seconds=0)
            ui.complete_budget_step(step)
            ui.log(f"Committed finding {step}/{budget}")
            seen.seen_steps.add(step)
        seen.n_attempts = 0
        ui.set_status(f"finding {len(seen.seen_steps)}/{budget}")

    if len(findings) >= budget or work_step > budget:
        return

    step_key = str(work_step)
    attempts = list((state.get("sql_attempts") or {}).get(step_key) or [])
    if len(attempts) > seen.n_attempts and len(findings) < work_step:
        last = attempts[-1]
        ui.set_budget_step(work_step)
        ui.set_status(f"SQL attempt {len(attempts)} (step {work_step}/{budget})")
        ui.set_sql_result(
            last.get("sql") or "",
            _execution_from_attempt(last),
        )
        seen.n_attempts = len(attempts)


@contextmanager
def _watch_opencode_lake_state(workdir: str, *, budget: int) -> Iterator[None]:
    """Poll lake_tool_state.json and refresh the Rich UI while OpenCode runs."""
    ui = get_ui()
    if not ui.rich_cli:
        yield
        return

    seen = _LakeUISeen()
    stop = threading.Event()

    def _poll() -> None:
        # Rich Live is updated from this thread while invoke_opencode blocks main.
        while not stop.is_set():
            try:
                _apply_lake_state_to_ui(workdir, ui, budget=budget, seen=seen)
            except (OSError, ValueError, TypeError, KeyError):
                pass
            stop.wait(DEFAULT_LAKE_UI_POLL_SEC)

    thread = threading.Thread(target=_poll, name="opencode-lake-ui", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2.0)
        _apply_lake_state_to_ui(workdir, ui, budget=budget, seen=seen)


def _budget_remaining(workdir: str) -> int:
    try:
        config = load_config(workdir)
    except FileNotFoundError:
        return 0
    budget = int(config["budget"])
    return max(0, budget - _findings_count(workdir))


def _max_continuation_rounds(workdir: str) -> int:
    try:
        config = load_config(workdir)
    except FileNotFoundError:
        return DEFAULT_MAX_OPENCODE_CONTINUATIONS
    budget = int(config["budget"])
    return max(budget * 2, DEFAULT_MAX_OPENCODE_CONTINUATIONS)


def _extract_agent_continuation_context(stdout: str, *, max_chars: int = 1500) -> str:
    """Return trailing stdout likely containing the agent's last prompt."""
    text = (stdout or "").strip()
    if not text:
        return "(no prior output captured)"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail_lines = lines[-12:] if len(lines) > 12 else lines
    context = "\n".join(tail_lines).strip()
    if len(context) > max_chars:
        context = "…\n" + context[-max_chars:]
    return context


def build_continuation_prompt(stdout: str) -> str:
    context = _extract_agent_continuation_context(stdout)
    return f"""No human is available. Proceed autonomously using your best judgment.

Your last output was:
{context}

Do not ask further questions or present option menus. Choose the best course of action yourself and continue executing sql/commit cycles until `status` reports the budget is exhausted."""


def build_resume_prompt(workdir: str) -> str:
    state = load_state(workdir)
    config = load_config(workdir)
    budget = int(config["budget"])
    n_findings = len(state.get("findings") or [])
    step = int(state.get("step") or 1)
    remaining = max(0, budget - n_findings)
    return f"""No human is available. Resume this research task autonomously.

Progress: {n_findings}/{budget} findings committed. Budget step is {step}. {remaining} finding(s) remain.

Findings already committed are recorded in lake_tool_state.json — do not redo them.
Continue from the current step with sql/commit cycles until `status` reports the budget is exhausted.

Do not ask questions or present option menus. Execute the next evidence-gathering step immediately."""


def _run_opencode_once(
    *,
    model: str,
    message: str,
    cwd: str,
    timeout_sec: int,
    session_id: Optional[str] = None,
    continue_session: bool = False,
    json_format: bool = False,
) -> subprocess.CompletedProcess[str]:
    # OpenCode blocks on inherited stdin when spawned non-interactively; use DEVNULL.
    cmd = ["opencode", "run", "--model", model, "--dir", cwd]
    if json_format:
        cmd.extend(["--format", "json"])
    if session_id:
        cmd.extend(["-s", session_id])
    elif continue_session:
        cmd.append("--continue")
    cmd.append(message)
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        stdin=subprocess.DEVNULL,
        env=_opencode_subprocess_env(cwd),
    )


def _attach_opencode_usage(
    meta: Dict[str, Any],
    *,
    cwd: str,
    silent: bool = False,
) -> None:
    xdg_data_home = _opencode_data_home(cwd)
    usage = fetch_all_opencode_session_usage(xdg_data_home=xdg_data_home)
    if usage:
        meta["usage"] = usage
        if meta.get("session_id") is None and usage.get("session_id"):
            meta["session_id"] = usage["session_id"]
        return
    if meta.get("session_id"):
        log(
            f"[opencode] Failed to fetch usage for session {meta['session_id']}.",
            silent=silent,
        )
        return
    log(
        "[opencode] No OpenCode session usage found in workspace DB.",
        silent=silent,
    )


def _finalize_opencode_meta(
    meta: Dict[str, Any],
    *,
    cwd: str,
    silent: bool = False,
) -> Dict[str, Any]:
    _attach_opencode_usage(meta, cwd=cwd, silent=silent)
    return meta


def invoke_opencode(
    *,
    model: str,
    prompt_path: str,
    cwd: str,
    timeout_sec: int = DEFAULT_OPENCODE_TIMEOUT_SEC,
    max_continuations: Optional[int] = None,
    silent: bool = False,
    resume_session_id: Optional[str] = None,
    resume_agent: bool = False,
) -> Dict[str, Any]:
    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read()

    max_rounds = max_continuations or _max_continuation_rounds(cwd)
    rounds: List[Dict[str, Any]] = []
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []

    resuming = resume_agent
    message = build_resume_prompt(cwd) if resuming else prompt
    session_id: Optional[str] = resume_session_id
    completed: Optional[subprocess.CompletedProcess[str]] = None
    logged_continue_fallback = False

    try:
        while len(rounds) < max_rounds:
            is_first_round = len(rounds) == 0
            json_format = session_id is None and is_first_round and not resuming
            continue_session = not is_first_round and session_id is None
            if continue_session and not logged_continue_fallback:
                log(
                    "[opencode] Session id not found in JSON output; using --continue.",
                    silent=silent,
                )
                logged_continue_fallback = True

            findings_before = _findings_count(cwd)
            completed = _run_opencode_once(
                model=model,
                message=message,
                cwd=cwd,
                timeout_sec=timeout_sec,
                session_id=session_id,
                continue_session=continue_session,
                json_format=json_format,
            )
            stdout_parts.append(completed.stdout or "")
            stderr_parts.append(completed.stderr or "")
            if session_id is None:
                session_id = parse_session_id_from_json_stdout(completed.stdout or "")
                if not is_first_round and session_id is None:
                    log(
                        "[opencode] Continuation round still missing session id.",
                        silent=silent,
                    )
            findings_after = _findings_count(cwd)
            rounds.append(
                {
                    "round": len(rounds) + 1,
                    "session_id": session_id,
                    "returncode": completed.returncode,
                    "findings_before": findings_before,
                    "findings_after": findings_after,
                    "budget_remaining": _budget_remaining(cwd),
                    "message_preview": message[:240],
                }
            )

            if completed.returncode != 0:
                break
            if _budget_remaining(cwd) <= 0:
                break
            if findings_after == findings_before and len(rounds) >= 2:
                prior = rounds[-2]
                if prior.get("findings_after") == findings_after:
                    break

            message = build_continuation_prompt(completed.stdout or "")
    except subprocess.TimeoutExpired as exc:
        stdout_parts.append(_coerce_subprocess_text(exc.stdout))
        stderr_parts.append(_coerce_subprocess_text(exc.stderr))
        if session_id is None:
            session_id = parse_session_id_from_json_stdout(
                _coerce_subprocess_text(exc.stdout)
            )
        rounds.append(
            {
                "round": len(rounds) + 1,
                "session_id": session_id,
                "returncode": None,
                "status": "timeout",
            }
        )
        return _finalize_opencode_meta(
            {
                "model": model,
                "returncode": None,
                "stdout": OPENCODE_ROUND_SEPARATOR.join(stdout_parts),
                "stderr": OPENCODE_ROUND_SEPARATOR.join(stderr_parts),
                "prompt_path": prompt_path,
                "status": "timeout",
                "rounds": rounds,
                "continuations": max(0, len(rounds) - 1),
                "session_id": session_id,
            },
            cwd=cwd,
            silent=silent,
        )

    assert completed is not None
    return _finalize_opencode_meta(
        {
            "model": model,
            "returncode": completed.returncode,
            "stdout": OPENCODE_ROUND_SEPARATOR.join(stdout_parts),
            "stderr": OPENCODE_ROUND_SEPARATOR.join(stderr_parts),
            "prompt_path": prompt_path,
            "status": "completed" if completed.returncode == 0 else "failed",
            "rounds": rounds,
            "continuations": max(0, len(rounds) - 1),
            "session_id": session_id,
        },
        cwd=cwd,
        silent=silent,
    )


def finalize_query_from_workspace(
    args: Namespace,
    query_record: Dict[str, Any],
    query_dir: str,
    *,
    topk_table_ids: List[str],
    topk_passage_ids: List[str],
    inference_clusters: List[dict],
    opencode_meta: Optional[Dict[str, Any]] = None,
    query_started: Optional[float] = None,
    usage_at_query_start=None,
    retained_clusters: int = 0,
    total_inference_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """Build pipeline-compatible result.json from lake tool state."""
    query_id = query_record["query_id"]
    user_query = query_record["query_text"]
    state = load_state(query_dir)
    findings = list(state.get("findings") or [])
    iterations = findings_to_iterations(findings)
    all_iterations = write_iteration_artifacts(query_dir, iterations, args.budget)

    assign_finding_indices(all_iterations)
    sync_finding_indices_to_query_dir(query_dir, all_iterations)
    report_iterations = [
        it for it in all_iterations if is_report_eligible_iteration(it)
    ]

    llm = get_llm_client(args.llm_provider, args.llm_model)
    report = generate_final_report(
        llm,
        user_query,
        report_iterations,
        args.temperature,
    )

    ui = get_ui()
    rich = ui.rich_cli
    metrics_tracker = _build_metrics_tracker(
        args,
        query_record=query_record,
        user_query=user_query,
        query_dir=query_dir,
        topk_table_ids=topk_table_ids,
        topk_passage_ids=topk_passage_ids,
        inference_clusters=inference_clusters,
        rich=rich,
        initial_candidate_clusters=retained_clusters,
    )
    _record_metrics(metrics_tracker, all_iterations, args.budget)

    query_usage = (
        get_tracker().snapshot().subtract(usage_at_query_start).to_dict()
        if usage_at_query_start is not None
        else {}
    )
    opencode_usage = (opencode_meta or {}).get("usage")
    query_time_taken = (
        round(time.perf_counter() - query_started, 3)
        if query_started is not None
        else 0.0
    )

    query_metrics = None
    if metrics_tracker:
        query_metrics = metrics_tracker.finalize(report, all_iterations)

    combined_usage = combine_query_usage(query_usage, opencode_usage)
    initial_retrieval_path = os.path.join(query_dir, INITIAL_RETRIEVAL_FILENAME)
    if not os.path.isfile(initial_retrieval_path):
        initial_retrieval_path = None
    initial_clusters_path = os.path.join(query_dir, INITIAL_CLUSTERS_FILENAME)
    if not os.path.isfile(initial_clusters_path):
        initial_clusters_path = None

    if total_inference_clusters is None:
        total_inference_clusters = len(inference_clusters)

    result = build_result_json(
        query_id=query_id,
        user_query=user_query,
        coverage=query_record.get("coverage"),
        method="opencode",
        answer=report,
        iterations=all_iterations,
        ground_truth=query_record.get("ground_truth"),
        topk_table_ids=topk_table_ids,
        topk_passage_ids=topk_passage_ids,
        total_inference_clusters=total_inference_clusters,
        retained_clusters=retained_clusters,
        time_taken=query_time_taken,
        usage=combined_usage,
        metrics=query_metrics,
        opencode_meta=opencode_meta,
        query_dir=query_dir,
        initial_retrieval_path=initial_retrieval_path,
        initial_clusters_path=initial_clusters_path,
    )
    save_query_result(
        query_dir,
        result,
        ground_truth=query_record.get("ground_truth"),
        topk_table_ids=topk_table_ids,
        topk_passage_ids=topk_passage_ids,
        metrics=query_metrics,
    )
    return result


def run_opencode_query(
    args: Namespace,
    query_record: Dict[str, Any],
    log_dir: str,
    sqlite_db_path: str,
    inference_clusters: List[dict],
    embedder=None,
    *,
    query_index: int = 1,
    skip_agent: bool = False,
) -> Dict[str, Any]:
    query_id = query_record["query_id"]
    user_query = query_record["query_text"]
    query_dir = ensure_dir(os.path.join(log_dir, str(query_id)))
    ui = get_ui()
    partial = partial_workspace_status(query_dir, args.budget)
    preserve_workspace = partial.resume_agent or partial.finalize_only

    ui.begin_query(query_index, query_id, user_query, args.budget)
    query_started = time.perf_counter()
    usage_at_query_start = get_tracker().snapshot()

    topk_table_ids: List[str] = []
    topk_passage_ids: List[str] = []
    initial_retrieval_path: Optional[str] = None
    initial_clusters_path: Optional[str] = None
    retained_clusters = 0
    total_inference_clusters = len(inference_clusters)
    metrics_inference_clusters = inference_clusters
    schema_descriptions_path = os.path.join(
        args.tables_lake_dir, "schema_descriptions.json"
    )
    passage_source_path = (
        resolve_passage_descriptions_source(args) if args.use_passages else None
    )

    if preserve_workspace:
        resume_ctx = _load_resume_workspace_context(query_dir)
        topk_table_ids = resume_ctx["topk_table_ids"]
        topk_passage_ids = resume_ctx["topk_passage_ids"]
        initial_retrieval_path = resume_ctx["initial_retrieval_path"]
        initial_clusters_path = resume_ctx["initial_clusters_path"]
        retained_clusters = resume_ctx["retained_clusters"]
        if resume_ctx["total_inference_clusters"]:
            total_inference_clusters = resume_ctx["total_inference_clusters"]
        if resume_ctx["metrics_inference_clusters"]:
            metrics_inference_clusters = resume_ctx["metrics_inference_clusters"]
        if partial.finalize_only:
            skip_agent = True
            log(
                f"[opencode] Finalizing query {query_id} from "
                f"{partial.n_findings}/{args.budget} committed findings.",
                silent=args.silent,
            )
        else:
            log(
                f"[opencode] Resuming query {query_id} from "
                f"{partial.n_findings}/{args.budget} committed findings.",
                silent=args.silent,
            )
    elif not _opencode_skip_retrieval(args):
        if ui.rich_cli:
            ui.set_status("relevance")
        if embedder is None:
            embedder = get_embedding_client(
                args.embedding_provider,
                args.embedding_model,
                gpu=args.gpu,
            )
        topk_table_ids, topk_passage_ids = _compute_relevance(
            args, user_query, embedder
        )
        initial_retrieval_path = write_initial_retrieval_artifact(
            query_dir,
            _build_initial_retrieval_payload(
                topk_table_ids,
                topk_passage_ids,
                schema_descriptions_path=schema_descriptions_path,
                passage_descriptions_path=passage_source_path,
            ),
        )
        if not _opencode_skip_clustering(args):
            clusters_payload = _build_initial_clusters_payload(
                inference_clusters,
                topk_table_ids,
                topk_passage_ids,
                args=args,
                passage_descriptions_path=passage_source_path,
            )
            initial_clusters_path = write_initial_clusters_artifact(
                query_dir,
                clusters_payload,
            )
            retained_clusters = clusters_payload["n_retained_clusters"]
            total_inference_clusters = clusters_payload["n_total_inference_clusters"]
            if not args.use_clustering:
                metrics_inference_clusters = clusters_payload["clusters"]
            if retained_clusters == 0:
                log(
                    "No inference clusters overlap top-k retrieval candidates.",
                    silent=args.silent,
                )

    _setup_query_workspace(
        args,
        query_dir,
        sqlite_db_path,
        reset_state=not preserve_workspace,
    )

    prompt_path = os.path.join(query_dir, "prompt.txt")
    if not preserve_workspace:
        prompt = build_opencode_prompt(
            query_id=query_id,
            user_query=user_query,
            schema_descriptions_path=schema_descriptions_path,
            sqlite_db_path=sqlite_db_path,
            query_dir=query_dir,
            budget=args.budget,
            max_sql_attempts=args.max_sql_attempts,
            use_passages=args.use_passages,
            passage_descriptions_path=(
                query_passage_descriptions_path(query_dir, args.passage_type)
                if args.use_passages
                else None
            ),
            initial_retrieval_path=initial_retrieval_path,
            initial_clusters_path=initial_clusters_path,
        )
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
    elif not os.path.isfile(prompt_path):
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("Resumed OpenCode run.\n")

    opencode_meta = None
    if not skip_agent:
        if ui.rich_cli:
            ui.set_status("Running OpenCode")
        model = f"{args.llm_provider}/{args.llm_model}"
        if not partial.resume_agent:
            log(
                f"[opencode] Running query {query_id} with model {model}",
                silent=args.silent,
            )
        resume_session_id: Optional[str] = None
        if partial.resume_agent:
            resume_session_id = load_latest_opencode_session_id(
                xdg_data_home=_opencode_data_home(query_dir),
            )
            if resume_session_id:
                log(
                    f"[opencode] Resuming OpenCode session {resume_session_id}.",
                    silent=args.silent,
                )
            else:
                log(
                    "[opencode] No OpenCode session id found; starting resume "
                    "without -s (may fall back to --continue).",
                    silent=args.silent,
                )
        try:
            with _watch_opencode_lake_state(query_dir, budget=args.budget):
                opencode_meta = invoke_opencode(
                    model=model,
                    prompt_path=prompt_path,
                    cwd=query_dir,
                    silent=args.silent,
                    resume_session_id=resume_session_id,
                    resume_agent=partial.resume_agent,
                )
        except subprocess.TimeoutExpired as exc:
            opencode_meta = {
                "model": model,
                "returncode": None,
                "stdout": _coerce_subprocess_text(exc.stdout),
                "stderr": _coerce_subprocess_text(exc.stderr),
                "status": "timeout",
            }
        stdout_path = os.path.join(query_dir, "opencode_stdout.txt")
        stderr_path = os.path.join(query_dir, "opencode_stderr.txt")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(opencode_meta.get("stdout") or "")
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(opencode_meta.get("stderr") or "")
        opencode_meta["stdout_path"] = stdout_path
        opencode_meta["stderr_path"] = stderr_path

    if ui.rich_cli:
        ui.set_status("finalizing")
    result = finalize_query_from_workspace(
        args,
        query_record,
        query_dir,
        topk_table_ids=topk_table_ids,
        topk_passage_ids=topk_passage_ids,
        inference_clusters=metrics_inference_clusters,
        opencode_meta=opencode_meta,
        query_started=query_started,
        usage_at_query_start=usage_at_query_start,
        retained_clusters=retained_clusters,
        total_inference_clusters=total_inference_clusters,
    )
    ui.set_final_result(
        result["answer"],
        query_id=query_id,
        n_iterations=len(result.get("findings") or []),
        result_path=os.path.join(query_dir, "result.json"),
        usage_summary=format_nested_usage_line(result_usage(result)),
    )
    return result


def run_opencode_job(
    args_dict: Dict[str, Any],
    query_record: Dict[str, Any],
    log_dir: str,
    sqlite_db_path: str,
    inference_clusters_path: str,
) -> Dict[str, Any]:
    """Submitit worker entrypoint: execute one OpenCode query in an isolated process."""
    from src.pipeline import args_from_dict

    args = args_from_dict(args_dict)
    set_seed(args.seed)
    init_ui(silent=True, rich_cli=False)
    reset_tracker()

    inference_clusters = load_opencode_inference_clusters(
        args,
        inference_clusters_path=inference_clusters_path,
    )

    query_dir = os.path.join(log_dir, str(query_record["query_id"]))
    partial = partial_workspace_status(query_dir, args.budget)
    preserve_workspace = partial.resume_agent or partial.finalize_only

    embedder = None
    if not _opencode_skip_retrieval(args) and not preserve_workspace:
        embedder = get_embedding_client(
            args.embedding_provider,
            args.embedding_model,
            gpu=args.gpu,
        )
    return run_opencode_query(
        args,
        query_record,
        log_dir,
        sqlite_db_path,
        inference_clusters,
        embedder,
    )


def run_opencode_submitit_pipeline(args: Namespace, log_dir: str) -> int:
    """Run OpenCode baseline with one submitit job per query."""
    from src.pipeline import save_run_queries
    from src.submitit_runner import (
        build_submitit_executor,
        report_merged_results,
        submit_and_wait_batched,
        worker_args,
    )

    submitit_folder = os.path.join(log_dir, "submitit_logs")
    os.makedirs(submitit_folder, exist_ok=True)

    print(f"Run directory: {log_dir}")
    print(f"Submitit logs: {submitit_folder}\n")

    sqlite_db_path = materialize_lake_sqlite(
        args.tables_lake_dir,
        os.path.join(log_dir, "data_lake.sqlite"),
    )
    print(f"Materialized data lake SQLite: {sqlite_db_path}")

    try:
        inference_clusters = setup_opencode_inference_clusters(args)
        _log_opencode_cluster_setup(args, inference_clusters, use_print=True)

        queries = resolve_queries(args)
        if len(queries) <= 1:
            print("Warning: only one query; --submitit adds overhead vs sequential mode")

        save_run_queries(log_dir, queries)

        args_dict = worker_args(args)
        clusters_path = args.inference_clusters_path
        executor = build_submitit_executor(submitit_folder, args)

        run_started = time.perf_counter()

        def submit_one(query: Dict[str, Any]) -> Tuple[str, Any]:
            job = executor.submit(
                run_opencode_job,
                args_dict,
                query,
                log_dir,
                sqlite_db_path,
                clusters_path,
            )
            return str(query["query_id"]), job

        completed, failures = submit_and_wait_batched(
            queries,
            local=bool(args.local),
            max_workers=args.max_workers,
            submit_one=submit_one,
        )

        if failures:
            print(
                f"\n{len(failures)} job(s) failed; merging {len(completed)} successful result(s).",
                file=sys.stderr,
            )
            if not completed:
                return 1
            completed_set = set(completed)
            queries = [q for q in queries if str(q["query_id"]) in completed_set]

        report_merged_results(
            log_dir,
            queries,
            compute_metrics=bool(args.compute_metrics),
            run_time_taken=time.perf_counter() - run_started,
        )
        return 1 if failures else 0
    finally:
        _remove_materialized_sqlite(sqlite_db_path)
        log(f"Removed materialized SQLite: {sqlite_db_path}", silent=args.silent)


def _remove_materialized_sqlite(sqlite_db_path: str) -> None:
    """Delete the on-disk lake DB and any SQLite journal sidecar files."""
    for suffix in ("", "-wal", "-shm"):
        path = f"{sqlite_db_path}{suffix}"
        if os.path.isfile(path):
            os.remove(path)


def run_opencode_baseline(args: Namespace, log_dir: str) -> str:
    """Run the OpenCode baseline and return results_all.json path."""
    from src.pipeline import (
        build_results_payload,
        print_run_completion,
        save_run_queries,
        write_results_all,
    )

    ui = init_ui(silent=args.silent, rich_cli=args.rich_cli)
    reset_tracker()
    run_started = time.perf_counter()

    with ui.session():
        if ui.rich_cli:
            ui.set_phase("Setup")

        sqlite_db_path = materialize_lake_sqlite(
            args.tables_lake_dir,
            os.path.join(log_dir, "data_lake.sqlite"),
        )
        log(f"Materialized data lake SQLite: {sqlite_db_path}", silent=args.silent)

        results: List[Dict[str, Any]] = []
        results_path = ""
        payload: Dict[str, Any] = {}

        try:
            inference_clusters = setup_opencode_inference_clusters(args)
            _log_opencode_cluster_setup(
                args,
                inference_clusters,
                silent=args.silent,
            )

            queries = resolve_queries(args)
            save_run_queries(log_dir, queries)
            if ui.rich_cli:
                ui.configure_queries(len(queries))
                ui.set_phase("OpenCode baseline")

            embedder = None
            if not _opencode_skip_retrieval(args):
                embedder = get_embedding_client(
                    args.embedding_provider,
                    args.embedding_model,
                    gpu=args.gpu,
                )

            if ui.rich_cli:
                for i, query in enumerate(queries, start=1):
                    results.append(
                        run_opencode_query(
                            args,
                            query,
                            log_dir,
                            sqlite_db_path,
                            inference_clusters,
                            embedder,
                            query_index=i,
                        )
                    )
                    ui.finish_query()
            else:
                for query in tqdm(queries, desc="Queries", disable=args.silent):
                    results.append(
                        run_opencode_query(
                            args,
                            query,
                            log_dir,
                            sqlite_db_path,
                            inference_clusters,
                            embedder,
                        )
                    )

            payload = build_results_payload(
                results,
                compute_metrics=bool(args.compute_metrics),
                run_time_taken=time.perf_counter() - run_started,
                usage=build_opencode_run_usage(results, get_tracker()),
                log_dir=log_dir,
            )
            results_path = write_results_all(log_dir, payload)
            ui.set_results_summary(
                results_path,
                n_queries=len(queries),
                n_completed=len(results),
            )
        finally:
            _remove_materialized_sqlite(sqlite_db_path)
            log(f"Removed materialized SQLite: {sqlite_db_path}", silent=args.silent)

    print_run_completion(
        n_queries=len(results),
        results_path=results_path,
        usage=UsageSummary(**payload["usage"]["total"]),
        metrics_summary=payload.get("metrics_summary"),
        silent=args.silent,
    )
    return results_path
