"""VERL reward function — FinQA exec_acc with Step 4 G(Q, D_s3) frozen generator.

Implements s3 paper §3 eq.4:
    GBR(Q) = Acc(G(Q, D_s3), A) - Acc(G(Q, D_RAG), A)

Where:
- Acc(·, A) is FinQA execution accuracy (numeric match via answer_normalizer)
- G is the frozen generator: Step 4 few-shot PoT pipeline (markdown chunking +
  exemplar selection + PoT prompt + Bedrock Kimi K2.5 + execute_program)
- D_s3 is the cumulative <important_info> chunks selected by the searcher's
  trajectory (parsed from solution_str)

Naive baseline term Acc(G(Q, D_RAG)) is precomputed and lives in
naive_correct.json. The returned reward is a clamped improvement indicator:
max(Acc(G(Q, D_s3), A) - Acc(G(Q, D_RAG), A), 0). On the hard-only training
split Acc(G(Q, D_RAG), A)=0, so this preserves the prior 0/1 reward scale while
matching the s3 improvement-over-naive intent.
"""

import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Step 4 + finqa_common are imported lazily to avoid module-load failures
# on machines that haven't installed them.
_LAZY_IMPORTED = False
_BEDROCK_CLIENT = None
_WEAVIATE_CLIENT = None
_INIT_LOCK = threading.Lock()
_STEP4_CACHE: Dict[str, Any] = {
    "train_norms": None,
    "train_exemplars": None,
    "train_doc_ids": None,
    "embedding_by_key": None,
}
_NAIVE_CACHE: Optional[Dict[str, bool]] = None


def _prepare_step4_cache(phaseb_inference) -> None:
    """Mirror Step 4 exemplar state into reward-local fast lookup structures.

    The original Step 4 code already preloads all train question embeddings into
    memory for similarity-based few-shot selection. During FinQA train
    naive_correct precompute, the query is itself a train question, so calling
    Titan embedding again for every item is redundant. We keep the same exemplar
    selection semantics but cache:

    - normalized train embedding matrix, avoiding per-call normalization;
    - doc_id::question -> embedding, avoiding per-train-question Titan calls.
    """
    state = getattr(phaseb_inference, "_fsp_state", {})
    train_embeddings = state.get("train_embeddings")
    train_exemplars = state.get("train_exemplars")
    train_doc_ids = state.get("train_doc_ids")
    if train_embeddings is None or train_exemplars is None or train_doc_ids is None:
        return
    if len(train_embeddings) == 0:
        return

    train_arr = np.asarray(train_embeddings)
    norms = np.linalg.norm(train_arr, axis=1, keepdims=True) + 1e-10
    embedding_by_key = {}
    for i, (doc_id, exemplar) in enumerate(zip(train_doc_ids, train_exemplars)):
        question = exemplar.get("question")
        if question:
            embedding_by_key[f"{doc_id}::{question}"] = train_arr[i]

    _STEP4_CACHE.update({
        "train_norms": train_arr / norms,
        "train_exemplars": train_exemplars,
        "train_doc_ids": train_doc_ids,
        "embedding_by_key": embedding_by_key,
    })


def _lazy_init():
    """One-shot import + client setup. Safe under concurrent reward calls."""
    global _LAZY_IMPORTED, _BEDROCK_CLIENT, _WEAVIATE_CLIENT
    if _LAZY_IMPORTED:
        return
    with _INIT_LOCK:
        if _LAZY_IMPORTED:
            return
        repo_root = Path(os.getenv(
            "FINQA_REPO_ROOT", "/home/ubuntu/fin-qa-research"
        ))
        sys.path.insert(0, str(repo_root / "src" / "finqa_common" / "src"))
        sys.path.insert(0, str(repo_root / "src" / "nr-pb-step-4-s3-integrated-policy" / "code"))
        sys.path.insert(0, str(repo_root / "src" / "nr-pb-step-4-s3-integrated-policy" / "train"))
        from finqa_common.utils import get_bedrock_client  # noqa: F401
        import weaviate  # noqa: F401
        import inference as phaseb_inference  # noqa: F401
        # Initialize FSP exemplar state once.
        try:
            phaseb_inference._load_fsp_state()
            _prepare_step4_cache(phaseb_inference)
        except Exception as e:
            print(f"[finqa_exec_acc] _load_fsp_state warning: {e}")
        _BEDROCK_CLIENT = get_bedrock_client(region=os.getenv("AWS_REGION", "us-east-1"))
        weaviate_url = os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080")
        hp = weaviate_url.replace("http://", "").replace("https://", "")
        host = hp.split(":")[0]
        port = int(hp.split(":")[1]) if ":" in hp else 8080
        _WEAVIATE_CLIENT = weaviate.connect_to_local(host=host, port=port)
        _LAZY_IMPORTED = True


# ── Trajectory parsing ─────────────────────────────────────────
_TAG_QUESTION = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_TAG_INFORMATION = re.compile(r"<information>(.*?)</information>", re.DOTALL)
_TAG_IMPORTANT = re.compile(r"<important_info>\s*\[([^\]]*)\]\s*</important_info>")
_DOC_BLOCK = re.compile(
    r"Doc (\d+)\(Title:\s*([^)]*)\)\s*(.*?)(?=\nDoc \d+\(Title:|\Z)",
    re.DOTALL,
)


def _parse_doc_block(info_body: str) -> List[Tuple[int, str, str]]:
    """Parse <information> body into [(doc_index, title, text), ...]."""
    return [
        (int(m.group(1)), m.group(2).strip(), m.group(3).strip())
        for m in _DOC_BLOCK.finditer(info_body)
    ]


def _extract_dse3_chunks(solution_str: str) -> List[Dict[str, Any]]:
    """Reconstruct D_s3 — the cumulative <important_info>-selected chunks.

    s3 paper §3 / Appendix B: <important_info>[indices]</important_info> tags
    apply to the most recent <information> block. We scan the trajectory in
    order, pair each information block with the subsequent important_info, and
    accumulate selected docs.
    """
    info_iter = list(_TAG_INFORMATION.finditer(solution_str))
    if not info_iter:
        return []
    selected: List[Dict[str, Any]] = []
    seen_keys = set()
    for i, info_m in enumerate(info_iter):
        info_body = info_m.group(1)
        info_end = info_m.end()
        scope_end = info_iter[i + 1].start() if i + 1 < len(info_iter) else len(solution_str)
        important_m = _TAG_IMPORTANT.search(solution_str[info_end:scope_end])
        if important_m is None:
            continue
        try:
            indices = {
                int(x.strip())
                for x in important_m.group(1).split(",")
                if x.strip().lstrip("-").isdigit()
            }
        except ValueError:
            indices = set()
        if not indices:
            continue
        for doc_idx, title, text in _parse_doc_block(info_body):
            if doc_idx in indices:
                key = (title, text[:60])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                selected.append({
                    "id": f"dse3_{len(selected)}_{doc_idx}",
                    "text": text,
                    "original_index": title,
                    "source_type": "dse3",
                })
    if not selected and info_iter:
        # Fallback: if the searcher never emitted important_info, use the
        # initial naive RAG top-k as D_s3 (paper §3.1 Begin-with-Search).
        for doc_idx, title, text in _parse_doc_block(info_iter[0].group(1)):
            selected.append({
                "id": f"naive_{doc_idx}",
                "text": text,
                "original_index": title,
                "source_type": "naive",
            })
    return selected


def _extract_question(solution_str: str) -> Optional[str]:
    m = _TAG_QUESTION.search(solution_str)
    if m is None:
        return None
    return m.group(1).strip()


def _extract_doc_id(solution_str: str) -> Optional[str]:
    m = re.search(r'doc_id["\']?\s*[:=]\s*["\']([^"\']+)["\']', solution_str)
    return m.group(1) if m else None


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def _candidate_naive_cache_paths() -> List[Path]:
    paths = []
    env_path = os.getenv("FINQA_NAIVE_CORRECT_PATH", "").strip()
    if env_path:
        paths.append(Path(env_path))
    repo_root = Path(os.getenv("FINQA_REPO_ROOT", "/home/ubuntu/fin-qa-research"))
    cwd = Path.cwd()
    paths.extend([
        cwd / "data" / "finqa_s3" / "naive_correct.json",
        cwd / "data" / "finqa_s3_full" / "naive_correct.json",
        cwd / "s3" / "data" / "finqa_s3" / "naive_correct.json",
        cwd / "s3" / "data" / "finqa_s3_full" / "naive_correct.json",
        repo_root.parent / "s3" / "data" / "finqa_s3" / "naive_correct.json",
        repo_root.parent / "s3" / "data" / "finqa_s3_full" / "naive_correct.json",
    ])
    out = []
    seen = set()
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def _load_naive_cache() -> Dict[str, bool]:
    global _NAIVE_CACHE
    if _NAIVE_CACHE is not None:
        return _NAIVE_CACHE
    for path in _candidate_naive_cache_paths():
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            _NAIVE_CACHE = {str(k): bool(v) for k, v in raw.items()}
            print(f"[finqa_exec_acc] loaded naive_correct cache: {path} ({len(_NAIVE_CACHE)} entries)")
            return _NAIVE_CACHE
    _NAIVE_CACHE = {}
    print("[finqa_exec_acc] naive_correct cache not found; using baseline=0")
    return _NAIVE_CACHE


def _naive_baseline_correct(
    question: str,
    doc_id: Optional[str],
    extra_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """Lookup Acc(G(Q, D_RAG), A) from naive_correct.json.

    Prefer doc_id from parquet extra_info, then doc_id embedded in the prompt.
    As a compatibility fallback for older parquet rows, match by question hash
    when the hash is unique in the cache.
    """
    if extra_info:
        doc_id = str(extra_info.get("doc_id") or doc_id or "")
        question = str(extra_info.get("question") or question)
    cache = _load_naive_cache()
    if not cache:
        return False

    qh = _question_hash(question)
    if doc_id:
        key = f"{doc_id}::{qh}"
        if key in cache:
            return bool(cache[key])

    suffix = f"::{qh}"
    matches = [bool(v) for k, v in cache.items() if k.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[finqa_exec_acc] ambiguous naive baseline for question hash {qh}; using baseline=0")
    return False


# ── Step 4 G(Q, D_s3) ──────────────────────────────────────────
def _run_step4_generator(
    question: str,
    chunks: List[Dict[str, Any]],
    exclude_doc_id: Optional[str],
) -> Optional[str]:
    """Run Step 4 few-shot PoT pipeline. Returns predicted answer string."""
    from finqa_common.utils import (
        execute_program,
        extract_number,
        extract_python_code,
        generate_answer,
        get_embedding,
    )
    import inference as phaseb_inference  # noqa: E402

    query_emb = _cached_train_query_embedding(question, exclude_doc_id)
    if query_emb is None:
        query_emb = get_embedding(question, "amazon.titan-embed-text-v2:0", _BEDROCK_CLIENT)
    if not query_emb:
        return None
    shot_number = int(os.getenv("FINQA_REWARD_SHOTS", "5"))
    exemplars = _select_exemplars_cached(query_emb, shot_number, exclude_doc_id)
    if exemplars is None:
        exemplars = phaseb_inference.select_exemplars_by_similarity(
            query_embedding=query_emb,
            shot_number=shot_number,
            exclude_doc_id=exclude_doc_id,
        )
    system_prompt, user_prompt = phaseb_inference.build_fewshot_program_prompt(
        question, chunks, exemplars
    )
    generated = generate_answer(
        user_prompt,
        os.getenv("FINQA_REWARD_GENERATOR", "moonshotai.kimi-k2.5"),
        _BEDROCK_CLIENT,
        int(os.getenv("FINQA_REWARD_MAX_TOKENS", "2048")),
        float(os.getenv("FINQA_REWARD_TEMP", "0.0")),
        system_prompt=system_prompt,
    )
    code = extract_python_code(generated)
    if code:
        predicted = execute_program(code)
        if str(predicted).upper().startswith("ERROR"):
            fallback = extract_number(generated)
            if fallback and fallback != generated.strip():
                predicted = fallback
    else:
        predicted = extract_number(generated)
    return predicted


def _cached_train_query_embedding(question: str, doc_id: Optional[str]) -> Optional[List[float]]:
    if not doc_id:
        return None
    embedding_by_key = _STEP4_CACHE.get("embedding_by_key") or {}
    emb = embedding_by_key.get(f"{doc_id}::{question}")
    if emb is None:
        return None
    return emb.tolist() if hasattr(emb, "tolist") else list(emb)


def _select_exemplars_cached(
    query_embedding: List[float],
    shot_number: int,
    exclude_doc_id: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    train_norms = _STEP4_CACHE.get("train_norms")
    train_exemplars = _STEP4_CACHE.get("train_exemplars")
    train_doc_ids = _STEP4_CACHE.get("train_doc_ids")
    if train_norms is None or train_exemplars is None or train_doc_ids is None:
        return None

    query_vec = np.asarray(query_embedding)
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    similarities = np.dot(train_norms, query_norm)
    if exclude_doc_id:
        similarities = similarities.copy()
        for i, doc_id in enumerate(train_doc_ids):
            if doc_id == exclude_doc_id:
                similarities[i] = -1.0
    top_indices = np.argsort(similarities)[-shot_number:][::-1]
    return [train_exemplars[i] for i in top_indices[::-1]]


# ── Public API ─────────────────────────────────────────────────
def compute_score_finqa(solution_str, ground_truth, method="strict",
                         format_score=0.0, score=1.0):
    """Simple-float-return variant. Calls Step 4 G internally."""
    s, _, _ = compute_score_finqa_rag(solution_str, ground_truth)
    return s


def compute_score_finqa_rag(
    solution_str,
    ground_truth,
    zeroshot_answers=None,
    data_source=None,
    use_utility_score=True,
    use_generation_score=True,
    extra_info=None,
):
    """VERL RewardManager-compatible scoring.

    Steps:
        1. Parse <question> and accumulate <important_info>-selected chunks
           into D_s3.
        2. Run Step 4 G(Q, D_s3) — exemplar-aware PoT, execute the generated
           code, extract numeric answer.
        3. Match predicted vs gold via finqa_common.answer_normalizer.

    Returns (score, None, None) matching rag_2.compute_score_rag signature.
    """
    try:
        _lazy_init()
    except Exception as e:
        print(f"[finqa_exec_acc] lazy_init error: {e}")
        return 0.0, None, None

    question = _extract_question(solution_str)
    if not question:
        return 0.0, None, None

    d_s3_chunks = _extract_dse3_chunks(solution_str)
    if not d_s3_chunks:
        return 0.0, None, None

    exclude_doc_id = _extract_doc_id(solution_str)
    if extra_info and not exclude_doc_id:
        maybe_doc_id = extra_info.get("doc_id")
        if maybe_doc_id:
            exclude_doc_id = str(maybe_doc_id)

    try:
        from finqa_common.answer_normalizer import check_answer_match, post_process_answer
    except Exception as e:
        print(f"[finqa_exec_acc] answer_normalizer import error: {e}")
        return 0.0, None, None

    try:
        predicted = _run_step4_generator(question, d_s3_chunks, exclude_doc_id)
    except Exception as e:
        print(f"[finqa_exec_acc] generator error: {e}")
        return 0.0, None, None

    if predicted is None:
        return 0.0, None, None

    try:
        predicted = post_process_answer(predicted, ground_truth)
        is_correct, _ = check_answer_match(predicted, ground_truth)
        generation_score = 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"[finqa_exec_acc] match error: {e}")
        generation_score = 0.0

    baseline_score = 1.0 if _naive_baseline_correct(question, exclude_doc_id, extra_info) else 0.0
    if use_utility_score:
        score = max(generation_score - baseline_score, 0.0)
    else:
        score = generation_score
    return score, None, baseline_score
