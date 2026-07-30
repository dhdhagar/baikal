"""Passage embedding generation for the HybridQA corpus."""

import json
import os
from typing import Dict, Iterator, List, Literal, Tuple

from src.embedding_client import get_embedding_client
from src.utils import load_json, log, save_json

PassageType = Literal["synth", "raw"]
PASSAGE_TYPES = ("synth", "raw")


def _iter_corpus_entries(corpus_path: str) -> Iterator[dict]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        start = f.read(1)
        f.seek(0)
        if start == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {corpus_path}.")
            yield from data
            return
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _title_from_raw_key(key: str) -> str:
    title = str(key or "").strip()
    if title.startswith("/wiki/"):
        title = title[len("/wiki/") :]
    return title.replace("_", " ").strip()


def _normalize_passage_record(record: dict, passage_type: PassageType) -> dict:
    uid = str(record.get("uid") or "").strip()
    if passage_type == "synth":
        return {
            "uid": uid,
            "title": record.get("title", ""),
            "text": record.get("text", ""),
            "passage_type": passage_type,
        }
    return {
        "uid": uid,
        "key": record.get("key", ""),
        "title": _title_from_raw_key(record.get("key", "")),
        "text": record.get("value", ""),
        "passage_type": passage_type,
    }


def build_passage_text(passage_info: dict) -> str:
    title = str(passage_info.get("title") or "").strip()
    text = str(passage_info.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def collect_passages(corpus_path: str, passage_type: PassageType = "synth") -> Dict[str, dict]:
    """Return uid -> normalized passage record (first occurrence wins)."""
    if passage_type not in PASSAGE_TYPES:
        raise ValueError(f"passage_type must be one of {PASSAGE_TYPES}; got {passage_type!r}")

    corpus_field = "synth_text" if passage_type == "synth" else "text"
    passages_by_uid: Dict[str, dict] = {}
    for entry in _iter_corpus_entries(corpus_path):
        for record in entry.get(corpus_field) or []:
            normalized = _normalize_passage_record(record, passage_type)
            uid = normalized["uid"]
            if uid and uid not in passages_by_uid:
                passages_by_uid[uid] = normalized
    if not passages_by_uid:
        raise ValueError(
            f"No {corpus_field} entries found in {corpus_path} for passage_type={passage_type!r}."
        )
    return passages_by_uid


def collect_synth_texts(corpus_path: str) -> Dict[str, dict]:
    """Backward-compatible alias for synth passages."""
    return collect_passages(corpus_path, passage_type="synth")


def build_passage_id_mapping(
    passages_by_uid: Dict[str, dict],
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """
    Assign stable P<id> labels sorted by uid.

    Returns (uid_to_passage_id, passage_descriptions keyed by passage_id).
    """
    uid_to_passage_id: Dict[str, str] = {}
    passage_descriptions: Dict[str, dict] = {}
    for idx, uid in enumerate(sorted(passages_by_uid), start=1):
        passage_id = f"P{idx}"
        uid_to_passage_id[uid] = passage_id
        record = passages_by_uid[uid]
        passage_descriptions[passage_id] = {
            "passage_id": passage_id,
            "uid": uid,
            "title": record.get("title", ""),
            "text": record.get("text", ""),
            "passage_type": record.get("passage_type", "synth"),
        }
        if record.get("passage_type") == "raw":
            passage_descriptions[passage_id]["key"] = record.get("key", "")
    return uid_to_passage_id, passage_descriptions


def _cached_passage_type(passage_descriptions: Dict[str, dict]) -> str | None:
    for record in passage_descriptions.values():
        return record.get("passage_type")
    return None


def ensure_passage_id_mapping(
    data_dir: str,
    corpus_path: str,
    mapping_path: str,
    descriptions_path: str,
    passage_type: PassageType = "synth",
    silent: bool = False,
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """Load or build uid->P<id> mapping and passage metadata."""
    if os.path.isfile(mapping_path) and os.path.isfile(descriptions_path):
        uid_to_passage_id = load_json(mapping_path)
        passage_descriptions = load_json(descriptions_path)
        cached_type = _cached_passage_type(passage_descriptions)
        if cached_type == passage_type:
            log(f"Using existing passage id mapping: {mapping_path}", silent=silent)
            return uid_to_passage_id, passage_descriptions
        log(
            f"Cached passage descriptions are type {cached_type!r}, "
            f"but passage_type={passage_type!r}; rebuilding mapping.",
            silent=silent,
        )

    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(
            f"Passage mapping not found at {mapping_path} and corpus not found at "
            f"{corpus_path}. Provide dpdisc_uid_to_passage_id.json or HybridQA_corpus.jsonl."
        )

    passages_by_uid = collect_passages(corpus_path, passage_type=passage_type)
    uid_to_passage_id, passage_descriptions = build_passage_id_mapping(passages_by_uid)
    save_json(mapping_path, uid_to_passage_id, indent=2)
    save_json(descriptions_path, passage_descriptions, indent=2)
    log(
        f"Wrote {passage_type} passage id mapping ({len(uid_to_passage_id)} passages) "
        f"-> {mapping_path}",
        silent=silent,
    )
    return uid_to_passage_id, passage_descriptions


def ensure_passage_embeddings(
    path: str,
    data_dir: str,
    corpus_path: str,
    mapping_path: str,
    descriptions_path: str,
    passage_type: PassageType = "synth",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedding_provider: str = "local",
    gpu: bool = False,
    silent: bool = False,
) -> str:
    """
    Return path to passage embeddings JSON.

    If the file already exists, skip generation. Otherwise embed title+text for each
    passage in the HybridQA corpus and write `{passage_id: embedding}`.
    """
    if os.path.isfile(path):
        log(f"Using existing passage embeddings: {path}", silent=silent)
        return path

    _, passage_descriptions = ensure_passage_id_mapping(
        data_dir,
        corpus_path,
        mapping_path,
        descriptions_path,
        passage_type=passage_type,
        silent=silent,
    )

    passage_ids = sorted(passage_descriptions.keys(), key=lambda pid: int(pid[1:]))
    docs = [build_passage_text(passage_descriptions[pid]) for pid in passage_ids]

    log(
        f"Generating embeddings for {len(passage_ids)} {passage_type} passages with "
        f"{embedding_provider}/{embedding_model!r}…",
        silent=silent,
    )

    embedder = get_embedding_client(embedding_provider, embedding_model, gpu=gpu)
    embs = embedder.encode(
        docs,
        batch_size=32,
        show_progress_bar=not silent,
        feature="passage_embedding_setup",
    )
    if hasattr(embs, "tolist"):
        embs = embs.tolist()
    if not isinstance(embs, list) or len(embs) != len(passage_ids):
        raise RuntimeError("Unexpected embeddings output from embedding client.")

    out: Dict[str, List[float]] = {
        pid: list(map(float, embs[i])) for i, pid in enumerate(passage_ids)
    }
    save_json(path, out, indent=2)
    log(f"Wrote passage embeddings -> {path}", silent=silent)
    return path
