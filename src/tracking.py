"""Token, cost, and latency tracking for OpenAI and LiteLLM API calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UsageSummary:
    """Aggregated token usage, cost, and latency."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    time_taken: float = 0.0
    n_calls: int = 0

    def add(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        cost_usd: float = 0.0,
        time_taken: float = 0.0,
        n_calls: int = 1,
    ) -> None:
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        if total_tokens is not None:
            self.total_tokens += int(total_tokens)
        else:
            self.total_tokens += int(prompt_tokens or 0) + int(completion_tokens or 0)
        self.cost_usd += float(cost_usd or 0.0)
        self.time_taken += float(time_taken or 0.0)
        self.n_calls += int(n_calls or 0)

    def merge(self, other: "UsageSummary") -> None:
        self.add(
            prompt_tokens=other.prompt_tokens,
            completion_tokens=other.completion_tokens,
            total_tokens=other.total_tokens,
            cost_usd=other.cost_usd,
            time_taken=other.time_taken,
            n_calls=other.n_calls,
        )

    def subtract(self, other: "UsageSummary") -> "UsageSummary":
        return UsageSummary(
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
            cost_usd=round(self.cost_usd - other.cost_usd, 8),
            time_taken=round(self.time_taken - other.time_taken, 6),
            n_calls=self.n_calls - other.n_calls,
        )

    def copy(self) -> "UsageSummary":
        return UsageSummary(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cost_usd=self.cost_usd,
            time_taken=self.time_taken,
            n_calls=self.n_calls,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "time_taken": round(self.time_taken, 3),
            "n_calls": self.n_calls,
        }


@dataclass
class UsageRecord:
    """One API call."""

    call_type: str
    feature: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    time_taken: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_type": self.call_type,
            "feature": self.feature,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "time_taken": round(self.time_taken, 3),
        }


@dataclass
class UsageTracker:
    """Session-scoped accumulator for API usage."""

    records: List[UsageRecord] = field(default_factory=list)
    total: UsageSummary = field(default_factory=UsageSummary)
    by_feature: Dict[str, UsageSummary] = field(default_factory=dict)
    by_call_type: Dict[str, UsageSummary] = field(default_factory=dict)

    def snapshot(self) -> UsageSummary:
        return self.total.copy()

    def record(
        self,
        *,
        call_type: str,
        feature: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: Optional[int] = None,
        cost_usd: float = 0.0,
        time_taken: float = 0.0,
    ) -> UsageRecord:
        feature = feature or "unknown"
        call_type = call_type or "unknown"
        total = (
            int(total_tokens)
            if total_tokens is not None
            else int(prompt_tokens or 0) + int(completion_tokens or 0)
        )
        cost = float(cost_usd or 0.0)
        elapsed = float(time_taken or 0.0)

        rec = UsageRecord(
            call_type=call_type,
            feature=feature,
            provider=provider,
            model=model,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=total,
            cost_usd=cost,
            time_taken=elapsed,
        )
        self.records.append(rec)

        self.total.add(
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            total_tokens=rec.total_tokens,
            cost_usd=rec.cost_usd,
            time_taken=rec.time_taken,
        )

        feat_summary = self.by_feature.setdefault(feature, UsageSummary())
        feat_summary.add(
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            total_tokens=rec.total_tokens,
            cost_usd=rec.cost_usd,
            time_taken=rec.time_taken,
        )

        type_summary = self.by_call_type.setdefault(call_type, UsageSummary())
        type_summary.add(
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            total_tokens=rec.total_tokens,
            cost_usd=rec.cost_usd,
            time_taken=rec.time_taken,
        )
        return rec

    def to_dict(self, *, include_records: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "total": self.total.to_dict(),
            "by_feature": {k: v.to_dict() for k, v in sorted(self.by_feature.items())},
            "by_call_type": {k: v.to_dict() for k, v in sorted(self.by_call_type.items())},
        }
        if include_records:
            out["records"] = [r.to_dict() for r in self.records]
        return out

    def format_cost(self) -> str:
        return f"${self.total.cost_usd:.4f}"

    def format_tokens(self) -> str:
        return f"{self.total.total_tokens:,} tokens"


_tracker: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker


def set_tracker(tracker: Optional[UsageTracker]) -> None:
    global _tracker
    _tracker = tracker


def reset_tracker() -> UsageTracker:
    tracker = UsageTracker()
    set_tracker(tracker)
    return tracker


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_usage(response: Any) -> Dict[str, int]:
    usage = _get_attr(response, "usage")
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(_get_attr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_get_attr(usage, "completion_tokens", 0) or 0)
    total_tokens = _get_attr(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    else:
        total_tokens = int(total_tokens or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _compute_cost_usd(
    *,
    response: Any,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    call_type: str,
) -> float:
    hidden = _get_attr(response, "_hidden_params") or {}
    if isinstance(hidden, dict):
        response_cost = hidden.get("response_cost")
        if response_cost is not None:
            return float(response_cost)

    try:
        import litellm
    except ImportError:
        return 0.0

    try:
        if call_type == "completion" and response is not None:
            return float(litellm.completion_cost(completion_response=response))
    except Exception:
        pass

    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(prompt_cost or 0.0) + float(completion_cost or 0.0)
    except Exception:
        return 0.0


def track_api_response(
    response: Any,
    *,
    provider: str,
    model: str,
    call_type: str,
    feature: str = "unknown",
    time_taken: float = 0.0,
) -> UsageRecord:
    """Record token usage, cost, and latency from an OpenAI or LiteLLM API response."""
    usage = _extract_usage(response)
    cost_usd = _compute_cost_usd(
        response=response,
        model=model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        call_type=call_type,
    )
    return get_tracker().record(
        call_type=call_type,
        feature=feature,
        provider=provider,
        model=model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        cost_usd=cost_usd,
        time_taken=time_taken,
    )


def format_usage_line(summary: UsageSummary) -> str:
    return (
        f"{summary.total_tokens:,} tokens "
        f"({summary.prompt_tokens:,} in / {summary.completion_tokens:,} out) "
        f"· ${summary.cost_usd:.4f} "
        f"· {summary.time_taken:.1f}s"
    )
