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
naive_correct.json. The smoke build uses the absolute accuracy Acc(G(Q, D_s3))
as the reward; subtract naive_correct in the trainer if/when the baseline term
is wired through.
"""

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Step 4 + finqa_common are imported lazily to avoid module-load failures
# on machines that haven't installed them.
_LAZY_IMPORTED = False
_BEDROCK_CLIENT = None
_WEAVIATE_CLIENT = None
_INIT_LOCK = threading.Lock()


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

    query_emb = get_embedding(question, "amazon.titan-embed-text-v2:0", _BEDROCK_CLIENT)
    if not query_emb:
        return None
    exemplars = phaseb_inference.select_exemplars_by_similarity(
        query_embedding=query_emb,
        shot_number=int(os.getenv("FINQA_REWARD_SHOTS", "5")),
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
        s = 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"[finqa_exec_acc] match error: {e}")
        s = 0.0

    return s, None, None
