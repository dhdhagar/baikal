"""LLM prior elicitation for cluster bandit selection (AutoDiscovery boolean_cat style)."""

from __future__ import annotations

from typing import Dict, Tuple

from src.cluster_selection import bandit_quality_label
from src.llm import LLMClient, chat_many, parse_json_dict_from_llm
from src.subquestions import format_cluster_context
from src.utils import log

UNINFORMED_PRIOR_ALPHA = 1.0
UNINFORMED_PRIOR_BETA = 1.0

CANNOT_COMMENT_KEY = "cannot_comment"

_SCORE_PER_CATEGORY = {
    "definitely_not": 0.0,
    "maybe_not": 0.25,
    "uncertain": 0.5,
    "maybe_yes": 0.75,
    "definitely_yes": 1.0,
}

_BELIEF_TO_COUNT_KEY = {
    "definitely yes": "definitely_yes",
    "maybe yes": "maybe_yes",
    "uncertain": "uncertain",
    "maybe not": "maybe_not",
    "definitely not": "definitely_not",
}

_COUNT_KEYS = tuple(_SCORE_PER_CATEGORY.keys())


def _empty_counts() -> Dict[str, int]:
    return {key: 0 for key in _COUNT_KEYS} | {CANNOT_COMMENT_KEY: 0}


def _scored_sample_count(counts: Dict[str, int]) -> int:
    return sum(counts.get(key, 0) for key in _COUNT_KEYS)


def build_cluster_prior_prompts(
    *,
    user_query: str,
    cluster: dict,
    bandit_reward: str,
) -> Tuple[str, str]:
    """Return (user prompt, system prompt) for cluster prior elicitation."""
    quality_label = bandit_quality_label(bandit_reward)
    cluster_id = cluster.get("cluster_id") or "unknown"
    prompt = f"""Assess whether the given cluster of tables and passages could potentially be useful to answer the following research question.

RESEARCH QUESTION:
{user_query}

CLUSTER OF TABLES AND PASSAGES (id={cluster_id}):
{format_cluster_context(cluster)}

Could this cluster be helpful to answer the research question, primarily when assessing {quality_label}?

Return ONLY a valid JSON:
{{"belief": "<one of: definitely yes, maybe yes, uncertain, maybe not, definitely not, cannot comment>"}}"""
    system_prompt = """\
You are a creative and insightful research scientist."""
    return prompt, system_prompt


def parse_belief_label(raw: str) -> str | None:
    """Parse one LLM belief sample; None if invalid."""
    try:
        data = parse_json_dict_from_llm(raw)
        belief = str(data.get("belief", "")).strip().lower()
    except ValueError:
        return None
    if belief == "cannot comment":
        return "cannot comment"
    return belief if belief in _BELIEF_TO_COUNT_KEY else None


def categorical_counts_to_beta_params(counts: Dict[str, int]) -> Tuple[float, float]:
    """Map categorical counts to Beta(alpha, beta) with uninformed prior (1, 1)."""
    frac_alpha = sum(
        counts.get(key, 0) * score for key, score in _SCORE_PER_CATEGORY.items()
    )
    n_valid = _scored_sample_count(counts)
    frac_beta = n_valid - frac_alpha
    return UNINFORMED_PRIOR_ALPHA + frac_alpha, UNINFORMED_PRIOR_BETA + frac_beta


def elicit_cluster_prior(
    llm: LLMClient,
    *,
    user_query: str,
    cluster: dict,
    bandit_reward: str,
    n_samples: int,
    temperature: float,
    silent: bool = False,
) -> Tuple[float, float, Dict[str, int], str, str]:
    """Sample LLM beliefs n times; return Beta prior, counts, and prompts."""
    cluster_id = cluster.get("cluster_id") or "unknown"
    prompt, system_prompt = build_cluster_prior_prompts(
        user_query=user_query,
        cluster=cluster,
        bandit_reward=bandit_reward,
    )
    raws = chat_many(
        llm,
        prompt,
        n=n_samples,
        system_prompt=system_prompt,
        temperature=temperature,
        feature="cluster_prior",
        json_mode=True,
    )
    counts = _empty_counts()
    for raw in raws:
        belief = parse_belief_label(raw)
        if belief is None:
            continue
        if belief == "cannot comment":
            counts[CANNOT_COMMENT_KEY] += 1
            continue
        count_key = _BELIEF_TO_COUNT_KEY[belief]
        counts[count_key] += 1

    if _scored_sample_count(counts) == 0:
        log(
            f"No valid LLM prior samples for cluster {cluster_id}; "
            "using uninformed prior.",
            silent=silent,
        )
        return UNINFORMED_PRIOR_ALPHA, UNINFORMED_PRIOR_BETA, counts, prompt, system_prompt

    alpha, beta = categorical_counts_to_beta_params(counts)
    return alpha, beta, counts, prompt, system_prompt
