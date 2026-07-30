"""Select which inference cluster to explore next."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scipy.stats import beta as beta_dist

from src.llm import LLMClient, chat, parse_json_dict_from_llm
from src.metrics.rubric_utils import extract_rubric_scores
from src.subquestions import format_cluster_context
from src.utils import log

BANDIT_REWARD_CHOICES = ("finding", "relevance", "distinctness", "usefulness")

_REWARD_COMPONENT_KEYS = {
    "relevance": "relevance",
    "distinctness": "distinctness",
    "usefulness": "report_usefulness",
}

_REWARD_LABELS = {
    "finding": "avg_finding_score",
    "relevance": "avg_grounded_relevance",
    "distinctness": "avg_grounded_distinctness",
    "usefulness": "avg_grounded_usefulness",
}


def bandit_quality_label(reward_kind: str) -> str:
    return _REWARD_LABELS[reward_kind]


@dataclass
class ClusterSelectionState:
    visits: Dict[str, int] = field(default_factory=dict)
    total_reward: Dict[str, float] = field(default_factory=dict)
    alpha: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.visits.clear()
        self.total_reward.clear()
        self.alpha.clear()
        self.beta.clear()

    def set_prior(self, cluster_id: str, alpha: float, beta: float) -> None:
        self.alpha[cluster_id] = alpha
        self.beta[cluster_id] = beta

    def posterior_mean(self, cluster_id: str) -> float:
        a = self.alpha.get(cluster_id, 0.5)
        b = self.beta.get(cluster_id, 0.5)
        return a / (a + b)

    def update_posterior(
        self, cluster_id: str, reward: float, *, weight: float = 1.0
    ) -> None:
        if cluster_id not in self.alpha:
            return
        self.alpha[cluster_id] += weight * reward
        self.beta[cluster_id] += weight * (1.0 - reward)

    def avg_reward(self, cluster_id: str) -> float:
        n = self.visits.get(cluster_id, 0)
        if n == 0:
            return 0.0
        return self.total_reward.get(cluster_id, 0.0) / n

    def total_visits(self) -> int:
        return sum(self.visits.values())

    def record_visit(self, cluster_id: str) -> int:
        self.visits[cluster_id] = self.visits.get(cluster_id, 0) + 1
        return self.visits[cluster_id]

    def record_outcome(self, cluster_id: str, reward: float) -> None:
        self.total_reward[cluster_id] = self.total_reward.get(cluster_id, 0.0) + reward


def bandit_reward_from_iteration(iteration: dict, reward_kind: str) -> float:
    rq = (iteration.get("metrics") or {}).get("research_quality") or {}
    rubric = rq.get("rubric")
    if reward_kind == "finding":
        return float(rq.get("finding_score") or 0.0)
    scores = extract_rubric_scores(rubric)
    grounded = float(scores.get("grounded") or 0.0)
    key = _REWARD_COMPONENT_KEYS[reward_kind]
    return grounded * float(scores.get(key) or 0.0)


def _cluster_id(cluster: dict) -> str:
    return cluster.get("cluster_id") or "unknown"


def _argmax_tiebreak(
    candidates: List[dict],
    score_fn,
    rng: random.Random,
) -> dict:
    best_score = -math.inf
    best: List[dict] = []
    for cluster in candidates:
        score = score_fn(cluster)
        if score > best_score:
            best_score = score
            best = [cluster]
        elif score == best_score:
            best.append(cluster)
    return rng.choice(best)


def _fallback_cluster(
    candidates: List[dict],
    rng: random.Random,
    reason: str,
    *,
    silent: bool,
) -> dict:
    log(
        f"Cluster LLM selection failed ({reason}); falling back to random.",
        silent=silent,
    )
    return rng.choice(candidates)


def _select_ucb(
    candidates: List[dict],
    state: ClusterSelectionState,
    rng: random.Random,
    ucb_c: float,
) -> dict:
    total = max(state.total_visits(), 1)

    def score_fn(cluster: dict) -> float:
        cid = _cluster_id(cluster)
        visits = state.visits.get(cid, 0)
        if visits == 0:
            return math.inf
        return state.avg_reward(cid) + ucb_c * math.sqrt(
            math.log(total) / visits
        )

    return _argmax_tiebreak(candidates, score_fn, rng)


def _select_bayes_ucb(
    candidates: List[dict],
    state: ClusterSelectionState,
    rng: random.Random,
) -> dict:
    t = max(state.total_visits() + 1, 2)
    q = 1.0 - 1.0 / t

    def score_fn(cluster: dict) -> float:
        cid = _cluster_id(cluster)
        return float(beta_dist.ppf(q, state.alpha[cid], state.beta[cid]))

    return _argmax_tiebreak(candidates, score_fn, rng)


def _select_epsilon_greedy(
    candidates: List[dict],
    state: ClusterSelectionState,
    rng: random.Random,
    epsilon: float,
) -> dict:
    if rng.random() < epsilon:
        return rng.choice(candidates)
    return _argmax_tiebreak(
        candidates, lambda c: state.avg_reward(_cluster_id(c)), rng
    )


def _select_epsilon_greedy_with_priors(
    candidates: List[dict],
    state: ClusterSelectionState,
    rng: random.Random,
    epsilon: float,
    tau: float,
) -> dict:
    if rng.random() < epsilon:
        weights = [
            math.exp(tau * state.posterior_mean(_cluster_id(c))) for c in candidates
        ]
        total = sum(weights)
        if total <= 0:
            return rng.choice(candidates)
        pick = rng.random() * total
        cumulative = 0.0
        for cluster, weight in zip(candidates, weights):
            cumulative += weight
            if pick <= cumulative:
                return cluster
        return candidates[-1]
    return _argmax_tiebreak(
        candidates, lambda c: state.posterior_mean(_cluster_id(c)), rng
    )


def _select_with_llm(
    llm: LLMClient,
    *,
    user_query: str,
    candidates: List[dict],
    state: ClusterSelectionState,
    bandit_reward: str,
    temperature: float,
    rng: random.Random,
    silent: bool,
) -> dict:
    reward_label = _REWARD_LABELS[bandit_reward]
    lines = []
    for i, cluster in enumerate(candidates, start=1):
        cid = _cluster_id(cluster)
        visits = state.visits.get(cid, 0)
        avg = state.avg_reward(cid) if visits else None
        avg_text = f"{avg:.3f}" if avg is not None else "n/a (never visited)"
        lines.append(
            f"{i}. cluster_id={cid}\n"
            f"   visits={visits}, {reward_label}={avg_text}\n"
            f"{format_cluster_context(cluster)}"
        )

    prompt = f"""You are helping build a deep-research report. Choose ONE candidate cluster to explore next.

RESEARCH QUESTION:
{user_query}

CANDIDATE CLUSTERS:
{chr(10).join(lines)}

Use your best judgment to select the next cluster to explore that may help discover a finding that improves on {reward_label}.

Return ONLY valid JSON:
{{"selected_index": <1-based index>}}"""

    raw = chat(
        llm,
        prompt,
        temperature=temperature,
        feature="cluster_selection",
        json_mode=True,
    )
    try:
        data = parse_json_dict_from_llm(raw)
        idx = int(data["selected_index"])
    except (ValueError, KeyError, TypeError) as exc:
        return _fallback_cluster(candidates, rng, str(exc), silent=silent)

    if 1 <= idx <= len(candidates):
        return candidates[idx - 1]
    return _fallback_cluster(
        candidates, rng, f"index {idx} out of range", silent=silent
    )


def select_cluster(
    candidate_clusters: List[dict],
    method: str,
    rng: random.Random,
    state: Optional[ClusterSelectionState] = None,
    *,
    llm: Optional[LLMClient] = None,
    user_query: str = "",
    temperature: float = 1.0,
    epsilon: float = 0.1,
    ucb_c: float = 1.0,
    bandit_reward: str = "finding",
    use_llm_priors: bool = False,
    llm_prior_tau: float = 1.0,
    silent: bool = False,
) -> dict:
    """Pick one cluster from candidates."""
    if not candidate_clusters:
        raise ValueError("No candidate clusters to select from.")

    if method == "random":
        return rng.choice(candidate_clusters)

    if state is None:
        raise ValueError(f"state is required for cluster_selection_method={method}")

    if method == "ucb":
        if use_llm_priors:
            return _select_bayes_ucb(candidate_clusters, state, rng)
        return _select_ucb(candidate_clusters, state, rng, ucb_c)

    if method == "epsilon-greedy":
        if use_llm_priors:
            return _select_epsilon_greedy_with_priors(
                candidate_clusters, state, rng, epsilon, llm_prior_tau
            )
        return _select_epsilon_greedy(candidate_clusters, state, rng, epsilon)

    if method == "llm":
        if llm is None:
            raise ValueError("llm is required for cluster_selection_method=llm")
        return _select_with_llm(
            llm,
            user_query=user_query,
            candidates=candidate_clusters,
            state=state,
            bandit_reward=bandit_reward,
            temperature=temperature,
            rng=rng,
            silent=silent,
        )

    raise ValueError(f"Unknown cluster_selection_method: {method}")
