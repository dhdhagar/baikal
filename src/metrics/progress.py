"""Progress reporting for metrics computation (inline UI or standalone Rich/tqdm)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Protocol, runtime_checkable

from src.cli_ui import get_ui


@runtime_checkable
class MetricsReporter(Protocol):
    def step_start(self, step: int, total: int, detail: str = "") -> None: ...

    def step_detail(self, detail: str) -> None: ...

    def step_done(self, step: int) -> None: ...

    def finish(self, summary: Optional[Dict[str, object]] = None) -> None: ...


class _MetricsActivityTracker:
    """Rolling state for metrics progress panels."""

    MAX_LOG = 8

    def __init__(self, total: int) -> None:
        self.total = total
        self.step = 0
        self.activity = "starting"
        self.log: List[str] = []

    def on_step_start(self, step: int, detail: str = "") -> None:
        self.step = step
        self.activity = detail or "computing"
        self._append(f"Step {step}: {self.activity}")

    def on_step_detail(self, detail: str) -> None:
        if not detail:
            return
        self.activity = detail
        if detail in ("LLM rubric", "diversity embedding") or detail.startswith("judge "):
            self._append(detail)

    def on_step_done(self, step: int) -> None:
        self._append(f"Step {step} complete")

    def on_finish(self, summary: Optional[Dict[str, object]] = None) -> None:
        if summary:
            report_score = summary.get("report_score")
            n_findings = summary.get("n_findings_valid")
            self.activity = f"complete · report_score={report_score}"
            self._append(
                f"Finished · report_score={report_score}, findings={n_findings}"
            )
        else:
            self.activity = "complete"

    def _append(self, message: str) -> None:
        self.log.append(message)
        self.log = self.log[-self.MAX_LOG :]

    def panel_body(self) -> str:
        lines = [
            f"Step      {self.step}/{self.total}",
            f"Activity  {self.activity or '—'}",
        ]
        if self.log:
            lines.extend(["", "Recent"])
            lines.extend(f"  • {entry}" for entry in self.log)
        return "\n".join(lines)


class NullMetricsReporter:
    def step_start(self, step: int, total: int, detail: str = "") -> None:
        del step, total, detail

    def step_detail(self, detail: str) -> None:
        del detail

    def step_done(self, step: int) -> None:
        del step

    def finish(self, summary: Optional[Dict[str, object]] = None) -> None:
        del summary


class UIMetricsReporter:
    """Update the active Rich dashboard during inline runs."""

    def __init__(self, total: int) -> None:
        self.total = total
        self._tracker = _MetricsActivityTracker(total)
        self._finished = False
        ui = get_ui()
        if ui.rich_cli:
            ui.begin_metrics(total)

    def _sync_ui(self) -> None:
        ui = get_ui()
        if ui.rich_cli:
            ui.set_metrics_panel(
                step=self._tracker.step,
                total=self.total,
                activity=self._tracker.activity,
                log=self._tracker.log,
            )
            return
        ui.set_status(
            f"metrics {self._tracker.step}/{self.total}: {self._tracker.activity}"
        )

    def step_start(self, step: int, total: int, detail: str = "") -> None:
        del total
        self._tracker.on_step_start(step, detail)
        self._sync_ui()

    def step_detail(self, detail: str) -> None:
        self._tracker.on_step_detail(detail)
        self._sync_ui()

    def step_done(self, step: int) -> None:
        self._tracker.on_step_done(step)
        self._sync_ui()

    def finish(self, summary: Optional[Dict[str, object]] = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._tracker.on_finish(summary)
        ui = get_ui()
        if ui.rich_cli:
            ui.set_metrics_panel(
                step=self._tracker.step,
                total=self.total,
                activity=self._tracker.activity,
                log=self._tracker.log,
            )
            ui.finish_metrics(summary)
            return
        if summary:
            ui.log(
                f"Metrics: report_score={summary.get('report_score')}, "
                f"findings={summary.get('n_findings_valid')}"
            )
        ui.set_status("metrics complete")


class TqdmMetricsReporter:
    def __init__(self, total: int, *, desc: str = "Metrics") -> None:
        from tqdm import tqdm

        self._finished = False
        self._tracker = _MetricsActivityTracker(total)
        self._bar = tqdm(total=total, desc=desc, leave=True)

    def step_start(self, step: int, total: int, detail: str = "") -> None:
        del total
        self._tracker.on_step_start(step, detail)
        self._bar.n = step - 1
        self._bar.set_postfix_str(self._tracker.activity[:48])
        self._bar.refresh()

    def step_detail(self, detail: str) -> None:
        self._tracker.on_step_detail(detail)
        if detail:
            self._bar.set_postfix_str(self._tracker.activity[:48])
            self._bar.refresh()

    def step_done(self, step: int) -> None:
        self._tracker.on_step_done(step)
        self._bar.n = step
        self._bar.refresh()

    def finish(self, summary: Optional[Dict[str, object]] = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._tracker.on_finish(summary)
        if summary:
            self._bar.set_postfix_str(
                f"report_score={summary.get('report_score')}",
                refresh=True,
            )
        self._bar.close()


class RichBarMetricsReporter:
    """Standalone Rich dashboard for offline metrics recomputation."""

    def __init__(self, total: int, *, desc: str = "Metrics") -> None:
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
        )
        from rich.rule import Rule

        self._finished = False
        self._tracker = _MetricsActivityTracker(total)
        self._rich_group = Group
        self._rich_live = Live
        self._rich_panel = Panel
        self._rich_rule = Rule
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[detail]}", style="dim"),
            auto_refresh=False,
            expand=True,
        )
        self._task = self._progress.add_task(desc, total=total, detail="starting")
        self._live = Live(
            self._render(),
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def _render_panel(self):
        return self._rich_panel(
            self._tracker.panel_body(),
            title="Metrics activity",
            border_style="magenta",
            padding=(0, 1),
        )

    def _render(self):
        return self._rich_group(
            self._progress,
            self._rich_rule(style="dim"),
            self._render_panel(),
        )

    def _refresh(self) -> None:
        self._live.update(self._render())

    def step_start(self, step: int, total: int, detail: str = "") -> None:
        del total
        self._tracker.on_step_start(step, detail)
        self._progress.update(
            self._task,
            completed=step - 1,
            detail=self._tracker.activity[:48],
        )
        self._refresh()

    def step_detail(self, detail: str) -> None:
        self._tracker.on_step_detail(detail)
        if detail:
            self._progress.update(self._task, detail=self._tracker.activity[:48])
            self._refresh()

    def step_done(self, step: int) -> None:
        self._tracker.on_step_done(step)
        self._progress.update(self._task, completed=step)
        self._refresh()

    def finish(self, summary: Optional[Dict[str, object]] = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._tracker.on_finish(summary)
        self._progress.update(
            self._task,
            completed=self._tracker.total,
            detail=self._tracker.activity[:48],
        )
        self._refresh()
        self._live.stop()


def make_metrics_reporter(
    budget: int,
    *,
    mode: str = "auto",
    silent: bool = False,
    rich_cli: bool = True,
    desc: str = "Metrics",
) -> MetricsReporter:
    if silent or budget <= 0:
        return NullMetricsReporter()
    ui = get_ui()
    if mode == "ui" or (mode == "auto" and ui.rich_cli):
        return UIMetricsReporter(budget)
    if rich_cli:
        return RichBarMetricsReporter(budget, desc=desc)
    return TqdmMetricsReporter(budget, desc=desc)


@contextmanager
def metrics_reporter(
    budget: int,
    *,
    mode: str = "bar",
    silent: bool = False,
    rich_cli: bool = True,
    desc: str = "Metrics",
) -> Iterator[MetricsReporter]:
    reporter = make_metrics_reporter(
        budget,
        mode=mode,
        silent=silent,
        rich_cli=rich_cli,
        desc=desc,
    )
    try:
        yield reporter
    finally:
        if hasattr(reporter, "_finished") and not reporter._finished:
            reporter.finish()
