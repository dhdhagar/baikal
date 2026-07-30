"""Terminal UI for the DPR discovery pipeline (standard print/tqdm or Rich live display)."""

from __future__ import annotations

import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

_ui: Optional["ConsoleUI"] = None


@runtime_checkable
class ConsoleUI(Protocol):
    silent: bool
    rich_cli: bool

    def log(self, msg: str, silent: bool = False) -> None: ...
    def session(self) -> Iterator["ConsoleUI"]: ...
    def set_phase(self, phase: str) -> None: ...
    def configure_queries(self, total: int) -> None: ...
    def begin_query(
        self, index: int, query_id: str, query_text: str, budget_total: int
    ) -> None: ...
    def finish_query(self) -> None: ...
    def complete_budget_step(self, step: int) -> None: ...
    def set_budget_step(self, step: int, detail: str = "") -> None: ...
    def set_cluster(self, cluster_id: str, description: str, meta: str = "") -> None: ...
    def set_cluster_pool_summary(self, retained: int, total: int) -> None: ...
    def set_sub_question(self, sub_question: str) -> None: ...
    def set_sql_result(
        self, sql: str = "", execution: Optional[Dict[str, Any]] = None
    ) -> None: ...
    def show_sql_answer(self, answer: str, *, pause_seconds: float = 2.0) -> None: ...
    def set_final_result(
        self,
        response: str,
        *,
        query_id: str = "",
        n_iterations: int = 0,
        result_path: str = "",
        usage_summary: str = "",
    ) -> None: ...
    def set_results_summary(
        self,
        results_path: str,
        *,
        n_queries: int = 0,
        n_completed: int = 0,
        usage_summary: str = "",
    ) -> None: ...
    def set_status(self, status: str) -> None: ...
    def begin_metrics(self, total: int) -> None: ...
    def set_metrics_panel(
        self,
        *,
        step: int,
        total: int,
        activity: str = "",
        log: Optional[List[str]] = None,
    ) -> None: ...
    def finish_metrics(self, summary: Optional[Dict[str, object]] = None) -> None: ...


def init_ui(*, silent: bool = False, rich_cli: bool = False) -> ConsoleUI:
    global _ui
    if rich_cli and not silent:
        _ui = RichConsoleUI(silent=silent)
    else:
        _ui = StandardConsoleUI(silent=silent)
    return _ui


def get_ui() -> ConsoleUI:
    return _ui if _ui is not None else StandardConsoleUI(silent=True)


def log(msg: str, silent: bool = False) -> None:
    get_ui().log(msg, silent=silent)


def format_sql_execution(
    execution: Optional[Dict[str, Any]],
    *,
    max_preview_rows: int = 5,
    max_cell_len: int = 48,
) -> str:
    """Compact text preview of SQL execution for the rich iteration panel."""
    if not execution:
        return ""
    if not execution.get("ok"):
        return f"Error: {execution.get('error') or 'unknown'}"
    row_count = int(execution.get("row_count", 0) or 0)
    rows = execution.get("rows") or []
    lines = [f"{row_count} row(s)"]
    for row in rows[:max_preview_rows]:
        if not isinstance(row, dict):
            lines.append(str(row)[:max_cell_len])
            continue
        parts = []
        for k, v in row.items():
            s = str(v) if v is not None else ""
            if len(s) > max_cell_len:
                s = s[: max_cell_len - 1] + "…"
            parts.append(f"{k}={s}")
        lines.append(" | ".join(parts))
    if row_count > len(rows[:max_preview_rows]):
        shown = min(len(rows), max_preview_rows)
        if row_count > shown:
            lines.append(f"… {row_count - shown} more row(s)")
    return "\n".join(lines)


class StandardConsoleUI:
    """Plain print logging and tqdm progress (used by default)."""

    rich_cli = False

    def __init__(self, silent: bool = False) -> None:
        self.silent = silent
        self._budget_total = 0
        self._log_sep = False

    @contextmanager
    def session(self) -> Iterator["StandardConsoleUI"]:
        yield self

    def log(self, msg: str, silent: bool = False) -> None:
        if silent or self.silent:
            return
        text = str(msg)
        if not text.strip():
            print()
            return
        stripped = text.lstrip()
        is_continuation = stripped.startswith(("  ", "\t")) or (
            len(stripped) > 2 and stripped[0].isdigit() and ". " in stripped[:4]
        )
        if self._log_sep and not is_continuation and not text.startswith("\n"):
            print()
        print(text)
        self._log_sep = True

    def set_phase(self, phase: str) -> None:
        pass

    def configure_queries(self, total: int) -> None:
        pass

    def begin_query(
        self,
        index: int,
        query_id: str,
        query_text: str,
        budget_total: int,
    ) -> None:
        if self.silent:
            return
        self._budget_total = budget_total
        preview = query_text if len(query_text) <= 120 else query_text[:120] + "..."
        print(f"\n=== Query {query_id} ===")
        print(preview)
        print()

    def finish_query(self) -> None:
        pass

    def complete_budget_step(self, step: int) -> None:
        pass

    def set_budget_step(self, step: int, detail: str = "") -> None:
        if self.silent:
            return
        suffix = f" — {detail}" if detail else ""
        total = self._budget_total or "?"
        self.log(f"\n--- Iteration {step}/{total}{suffix} ---")

    def set_cluster(self, cluster_id: str, description: str, meta: str = "") -> None:
        pass

    def set_cluster_pool_summary(self, retained: int, total: int) -> None:
        if self.silent:
            return
        self.log(
            f"Retained {retained}/{total} clusters after relevance filtering "
            f"for the sub-question loop."
        )

    def set_sub_question(self, sub_question: str) -> None:
        self.log(f"Sub-question:\n  {sub_question.strip()}")

    def set_sql_result(
        self, sql: str = "", execution: Optional[Dict[str, Any]] = None
    ) -> None:
        if sql.strip():
            self.log(f"SQL:\n{sql.strip()}")
        if execution is not None:
            self.log(f"Output:\n{format_sql_execution(execution)}")

    def show_sql_answer(self, answer: str, *, pause_seconds: float = 2.0) -> None:
        if (answer or "").strip():
            self.log(f"Answer:\n{answer.strip()}")

    def set_final_result(
        self,
        response: str,
        *,
        query_id: str = "",
        n_iterations: int = 0,
        result_path: str = "",
        usage_summary: str = "",
    ) -> None:
        meta = []
        if query_id:
            meta.append(f"Query: {query_id}")
        if n_iterations:
            meta.append(f"Iterations: {n_iterations}")
        if usage_summary:
            meta.append(f"Usage: {usage_summary}")
        if result_path:
            meta.append(f"Saved: {result_path}")
        if meta:
            self.log("\n".join(meta))
        if (response or "").strip():
            self.log(f"\nFinal response:\n{response.strip()}")

    def set_results_summary(
        self,
        results_path: str,
        *,
        n_queries: int = 0,
        n_completed: int = 0,
        usage_summary: str = "",
    ) -> None:
        lines = [f"Results file: {results_path}"]
        if n_queries:
            lines.append(f"Queries completed: {n_completed}/{n_queries}")
        if usage_summary:
            lines.append(f"Run usage: {usage_summary}")
        self.log("\n" + "\n".join(lines))

    def set_status(self, status: str) -> None:
        pass

    def begin_metrics(self, total: int) -> None:
        del total

    def set_metrics_panel(
        self,
        *,
        step: int,
        total: int,
        activity: str = "",
        log: Optional[List[str]] = None,
    ) -> None:
        del step, total, activity, log

    def finish_metrics(self, summary: Optional[Dict[str, object]] = None) -> None:
        del summary


@dataclass
class _ViewState:
    phase: str = "Starting"
    query_index: int = 0
    query_total: int = 0
    query_id: str = ""
    query_text: str = ""
    budget_current: int = 0
    budget_total: int = 0
    cluster_id: str = ""
    cluster_desc: str = ""
    cluster_meta: str = ""
    cluster_pool_summary: str = ""
    sub_question: str = ""
    sql_query: str = ""
    sql_output: str = ""
    sql_answer: str = ""
    final_response: str = ""
    final_meta: str = ""
    usage_summary: str = ""
    results_summary: str = ""
    status: str = ""
    metrics_active: bool = False
    metrics_step: int = 0
    metrics_total: int = 0
    metrics_activity: str = ""
    metrics_log: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


class RichConsoleUI:
    """Live dashboard: dual progress bars + current-iteration panel + recent log lines."""

    rich_cli = True
    _MAX_MESSAGES = 10
    _WRAP = 78
    _MAX_FINAL_RESPONSE = 2000

    def __init__(self, silent: bool = False) -> None:
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
        from rich.rule import Rule
        from rich.text import Text

        self._rich_group = Group
        self._rich_live = Live
        self._rich_panel = Panel
        self._rich_progress = Progress
        self._rich_bar = BarColumn
        self._rich_task_col = TaskProgressColumn
        self._rich_text_col = TextColumn
        self._rich_rule = Rule
        self._rich_text = Text

        self.silent = silent
        self._state = _ViewState()
        self._live: Optional[Live] = None
        self._progress = self._new_progress()
        self._query_task: Optional[int] = None
        self._budget_task: Optional[int] = None

    def _new_progress(self, *, console=None):
        # Parent Live drives refresh; Progress must not auto-refresh on its own.
        return self._rich_progress(
            self._rich_text_col("[bold]{task.description}"),
            self._rich_bar(bar_width=36),
            self._rich_task_col(),
            self._rich_text_col("•"),
            self._rich_text_col("{task.fields[detail]}", style="dim"),
            console=console,
            auto_refresh=False,
            expand=True,
        )

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()

    @contextmanager
    def session(self) -> Iterator["RichConsoleUI"]:
        if self.silent:
            yield self
            return
        with self._rich_live(
            self._render(),
            refresh_per_second=8,
            transient=False,
        ) as live:
            self._live = live
            self._stop_progress()
            self._progress = self._new_progress(console=live.console)
            self._query_task = None
            self._budget_task = None
            try:
                yield self
            finally:
                self._stop_progress()
                self._live = None

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _truncate(self, text: str, max_len: int = 72) -> str:
        text = " ".join((text or "").split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    def _wrap_block(self, label: str, value: str, indent: str = "  ") -> str:
        if not value:
            return ""
        prefix = f"{indent}{label}: "
        lines = textwrap.wrap(
            value,
            width=self._WRAP - len(indent),
            initial_indent=prefix,
            subsequent_indent=indent + " " * (len(label) + 2),
        )
        return "\n".join(lines)

    def _append_code_block(self, lines: List[str], label: str, value: str) -> None:
        if not value:
            return
        block = self._wrap_block(label, value)
        if block:
            lines.append(block)

    def _append_multiline_block(
        self, lines: List[str], label: str, value: str, indent: str = "  "
    ) -> None:
        if not value:
            return
        sub_indent = indent + " " * (len(label) + 2)
        lines.append(f"{indent}{label}:")
        for part in value.splitlines():
            wrapped = textwrap.wrap(
                part,
                width=self._WRAP - len(sub_indent),
                initial_indent=sub_indent,
                subsequent_indent=sub_indent,
            )
            lines.extend(wrapped or [sub_indent + part])

    def log(self, msg: str, silent: bool = False) -> None:
        if silent or self.silent:
            return
        text = str(msg).strip()
        if not text:
            return
        self._state.messages.append(text)
        self._state.messages = self._state.messages[-self._MAX_MESSAGES :]
        self._refresh()

    def set_phase(self, phase: str) -> None:
        self._state.phase = phase
        self._refresh()

    def configure_queries(self, total: int) -> None:
        if self.silent:
            return
        self._state.query_total = max(total, 0)
        self._stop_progress()
        console = self._live.console if self._live is not None else None
        self._progress = self._new_progress(console=console)
        self._query_task = self._progress.add_task(
            "Queries",
            total=total or 1,
            detail="",
        )
        self._budget_task = self._progress.add_task(
            "Budget",
            total=1,
            detail="idle",
        )
        self._refresh()

    def begin_query(
        self,
        index: int,
        query_id: str,
        query_text: str,
        budget_total: int,
    ) -> None:
        self._state.query_index = index
        self._state.query_id = query_id
        self._state.query_text = query_text
        self._state.budget_total = budget_total
        self._state.budget_current = 0
        self._state.cluster_id = ""
        self._state.cluster_desc = ""
        self._state.cluster_meta = ""
        self._state.cluster_pool_summary = ""
        self._state.sub_question = ""
        self._state.sql_query = ""
        self._state.sql_output = ""
        self._state.sql_answer = ""
        self._state.final_response = ""
        self._state.final_meta = ""
        self._state.usage_summary = ""
        self._state.status = ""
        self._state.phase = f"Query {query_id}"
        if not self.silent and self._query_task is not None:
            self._progress.update(
                self._query_task,
                completed=index - 1,
                description=f"Queries ({index}/{self._state.query_total})",
                detail=self._truncate(query_id, 40),
            )
        if not self.silent and self._budget_task is not None:
            self._progress.update(
                self._budget_task,
                total=max(budget_total, 1),
                completed=0,
                description="Budget",
                detail="starting",
            )
        self.log(f"Query {index}/{self._state.query_total}: {query_id}")
        self._refresh()

    def finish_query(self) -> None:
        if not self.silent and self._query_task is not None:
            self._progress.update(
                self._query_task,
                completed=self._state.query_index,
            )
        self._refresh()

    def complete_budget_step(self, step: int) -> None:
        if not self.silent and self._budget_task is not None:
            self._progress.update(self._budget_task, completed=step)
        self._refresh()

    def set_budget_step(self, step: int, detail: str = "") -> None:
        self._state.budget_current = step
        self._state.phase = f"Iteration {step}/{self._state.budget_total}"
        self._state.cluster_id = ""
        self._state.cluster_desc = ""
        self._state.cluster_meta = ""
        self._state.sub_question = ""
        self._state.sql_query = ""
        self._state.sql_output = ""
        self._state.sql_answer = ""
        self._state.status = detail or "selecting cluster"
        if not self.silent and self._budget_task is not None:
            self._progress.update(
                self._budget_task,
                completed=step - 1,
                description=f"Budget ({step}/{self._state.budget_total})",
                detail=self._truncate(detail or "working", 36),
            )
        self._refresh()

    def set_cluster(
        self,
        cluster_id: str,
        description: str,
        meta: str = "",
    ) -> None:
        self._state.cluster_id = cluster_id
        self._state.cluster_desc = description
        self._state.cluster_meta = meta
        self._state.status = "generating sub-questions"
        self._refresh()

    def set_cluster_pool_summary(self, retained: int, total: int) -> None:
        self._state.cluster_pool_summary = (
            f"{retained}/{total} clusters retained for sub-question loop"
        )
        if not self.silent:
            self.log(self._state.cluster_pool_summary)
        self._refresh()

    def set_sub_question(self, sub_question: str) -> None:
        self._state.sub_question = sub_question
        self._state.sql_query = ""
        self._state.sql_output = ""
        self._state.sql_answer = ""
        self._state.status = "SQL loop"
        self._refresh()

    def set_sql_result(
        self, sql: str = "", execution: Optional[Dict[str, Any]] = None
    ) -> None:
        if sql:
            self._state.sql_query = sql.strip()
        if execution is not None:
            self._state.sql_output = format_sql_execution(execution)
        self._state.status = "SQL loop"
        self._refresh()

    def show_sql_answer(self, answer: str, *, pause_seconds: float = 2.0) -> None:
        self._state.sql_answer = (answer or "").strip()
        self._state.status = "answer"
        self._refresh()
        if self.silent or pause_seconds <= 0:
            return
        time.sleep(pause_seconds)

    def set_final_result(
        self,
        response: str,
        *,
        query_id: str = "",
        n_iterations: int = 0,
        result_path: str = "",
        usage_summary: str = "",
    ) -> None:
        text = (response or "").strip()
        if len(text) > self._MAX_FINAL_RESPONSE:
            text = text[: self._MAX_FINAL_RESPONSE - 1] + "…"
        self._state.final_response = text
        meta_parts: List[str] = []
        if query_id:
            meta_parts.append(f"Query: {query_id}")
        if n_iterations:
            meta_parts.append(f"Iterations: {n_iterations}")
        if usage_summary:
            meta_parts.append(f"Usage: {usage_summary}")
        if result_path:
            meta_parts.append(f"Saved: {result_path}")
        self._state.final_meta = "  ·  ".join(meta_parts)
        self._state.usage_summary = usage_summary
        self._state.phase = "Final response"
        self._state.status = "complete"
        self._refresh()

    def set_results_summary(
        self,
        results_path: str,
        *,
        n_queries: int = 0,
        n_completed: int = 0,
        usage_summary: str = "",
    ) -> None:
        lines = [f"File: {results_path}"]
        if n_queries:
            lines.append(f"Queries completed: {n_completed}/{n_queries}")
        if usage_summary:
            lines.append(f"Run usage: {usage_summary}")
        self._state.results_summary = "\n".join(lines)
        if usage_summary:
            self._state.usage_summary = usage_summary
        self._state.phase = "Done"
        self._refresh()

    def set_status(self, status: str) -> None:
        self._state.status = status
        if not self.silent and self._budget_task is not None:
            self._progress.update(
                self._budget_task,
                detail=self._truncate(status, 36),
            )
        self._refresh()

    def begin_metrics(self, total: int) -> None:
        self._state.metrics_active = True
        self._state.metrics_step = 0
        self._state.metrics_total = max(total, 0)
        self._state.metrics_activity = "starting"
        self._state.metrics_log = []
        self._refresh()

    def set_metrics_panel(
        self,
        *,
        step: int,
        total: int,
        activity: str = "",
        log: Optional[List[str]] = None,
    ) -> None:
        self._state.metrics_active = True
        self._state.metrics_step = step
        self._state.metrics_total = max(total, 0)
        self._state.metrics_activity = activity
        if log is not None:
            self._state.metrics_log = list(log)
        self._state.status = activity or self._state.status
        if not self.silent and self._budget_task is not None:
            self._progress.update(
                self._budget_task,
                detail=self._truncate(activity or "metrics", 36),
            )
        self._refresh()

    def finish_metrics(self, summary: Optional[Dict[str, object]] = None) -> None:
        if summary:
            self.log(
                f"Metrics: report_score={summary.get('report_score')}, "
                f"findings={summary.get('n_findings_valid')}"
            )
        self._state.metrics_active = False
        self._state.status = "metrics complete"
        self._refresh()

    def _render_iteration_panel(self):
        s = self._state
        lines: List[str] = []
        lines.append(f"Phase     {s.phase}")
        if s.query_id:
            lines.append(f"Query     {s.query_id}")
            qline = self._wrap_block("Text", s.query_text)
            if qline:
                lines.append(qline)
        if s.budget_total:
            lines.append(
                f"Step      {s.budget_current}/{s.budget_total}  ·  {s.status or '—'}"
            )
        if s.cluster_pool_summary:
            lines.append(f"Clusters  {s.cluster_pool_summary}")
        if s.cluster_id:
            lines.append(f"Cluster   {s.cluster_id}")
            dline = self._wrap_block("Desc", s.cluster_desc)
            if dline:
                lines.append(dline)
            if s.cluster_meta:
                lines.append(f"          {s.cluster_meta}")
        if s.sub_question:
            sq = self._wrap_block("Sub-Q", s.sub_question)
            if sq:
                lines.append(sq)
        if s.sql_query:
            self._append_code_block(lines, "SQL", s.sql_query)
        if s.sql_output:
            self._append_multiline_block(lines, "Output", s.sql_output)
        if s.sql_answer:
            self._append_multiline_block(lines, "Answer", s.sql_answer)
        body = "\n".join(lines) if lines else "Waiting…"
        return self._rich_panel(
            body,
            title="Current iteration",
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_metrics_panel(self):
        s = self._state
        if not s.metrics_active:
            return None
        lines = [
            f"Step      {s.metrics_step}/{s.metrics_total}",
            f"Activity  {s.metrics_activity or '—'}",
        ]
        if s.metrics_log:
            lines.extend(["", "Recent"])
            lines.extend(f"  • {entry}" for entry in s.metrics_log)
        return self._rich_panel(
            "\n".join(lines),
            title="Metrics activity",
            border_style="magenta",
            padding=(0, 1),
        )

    def _render_final_panel(self):
        s = self._state
        if not s.final_response and not s.results_summary:
            return None
        lines: List[str] = []
        if s.final_meta:
            lines.append(s.final_meta)
        if s.usage_summary and "Usage:" not in (s.final_meta or ""):
            lines.append(f"Usage: {s.usage_summary}")
        if s.final_response:
            if lines:
                lines.append("")
            self._append_multiline_block(lines, "Response", s.final_response)
        if s.results_summary:
            if lines:
                lines.append("")
            self._append_multiline_block(lines, "Results", s.results_summary)
        body = "\n".join(lines)
        return self._rich_panel(
            body,
            title="Final response & results",
            border_style="green",
            padding=(0, 1),
        )

    def _render_messages(self):
        if not self._state.messages:
            return self._rich_text("No messages yet.", style="dim")
        return self._rich_text("\n\n".join(self._state.messages), style="dim")

    def _render(self):
        header = self._rich_text("DPR Discovery", style="bold white on blue")
        parts = [
            header,
            self._rich_rule(style="blue"),
            self._progress,
            self._rich_rule(style="dim"),
            self._render_iteration_panel(),
        ]
        metrics_panel = self._render_metrics_panel()
        if metrics_panel is not None:
            parts.extend([self._rich_rule(style="dim"), metrics_panel])
        final_panel = self._render_final_panel()
        if final_panel is not None:
            parts.extend([self._rich_rule(style="dim"), final_panel])
        parts.extend(
            [
                self._rich_rule(style="dim"),
                self._rich_panel(
                    self._render_messages(),
                    title="Log",
                    border_style="dim",
                    padding=(0, 1),
                ),
            ]
        )
        return self._rich_group(*parts)
