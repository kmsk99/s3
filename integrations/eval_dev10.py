"""dev 10 evaluation harness for our PPO-trained s3 search agent.

Runs the s3 paper Algorithm 1 multi-turn loop with:
- Searcher: our PPO'd Qwen2.5-7B-Instruct via vLLM (OpenAI API on EC2)
- Retriever: finqa_retrieval_adapter at http://127.0.0.1:3000/retrieve
- Frozen generator G: Step 4 PoT via phaseb_inference (Bedrock Kimi K2.5)
- Scoring: finqa_common.answer_normalizer (FinQA exec_acc)

Outputs per-question + summary scores comparing baseline (D_RAG only) vs s3
trained policy (D_s3 from multi-turn rollout).

Usage:
    python eval_dev10.py \\
        --finqa-dir .../dataset \\
        --searcher-url http://127.0.0.1:8001/v1/chat/completions \\
        --retrieval-url http://127.0.0.1:3000/retrieve \\
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# s3 paper grammar parsers (mirrors generation_s3.py)
_TAG_QUERY = re.compile(r"<query>(.*?)</query>", re.DOTALL)
_TAG_IMPORTANT = re.compile(r"<important_info>\s*\[([^\]]*)\]\s*</important_info>")
_TAG_COMPLETE = re.compile(r"<search_complete>(.*?)</search_complete>", re.DOTALL)

# s3 paper system prompt
S3_SYSTEM_PROMPT = (
    "You are a search copilot for a generation model. Based on a user's query "
    "and initial searched results, you will first determine if the searched "
    "results are enough to produce an answer. If the searched results are "
    "enough, you will use <search_complete>True</search_complete> to indicate "
    "that you have gathered enough information for the generation model to "
    "produce an answer.\n\n"
    "If the searched results are not enough, you will go through a loop of "
    "<query> → <information> → <important_info> → <search_complete> → <query> "
    "(if not complete) ..., to help the generation model to generate a better "
    "answer with more relevant information searched.\n\n"
    "Strict output grammar each turn:\n"
    "  <query>{\"query\": \"...\"}</query>\n"
    "  <important_info>[doc_index, ...]</important_info>\n"
    "  <search_complete>True|False</search_complete>\n"
)


def parse_query(text: str) -> str:
    m = _TAG_QUERY.search(text or "")
    if not m:
        return ""
    body = m.group(1).strip()
    try:
        return json.loads(body).get("query", "").strip()
    except Exception:
        return body


def parse_important(text: str) -> List[int]:
    m = _TAG_IMPORTANT.search(text or "")
    if not m:
        return []
    try:
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip().lstrip("-").isdigit()]
    except Exception:
        return []


def parse_complete(text: str) -> Optional[bool]:
    m = _TAG_COMPLETE.search(text or "")
    if not m:
        return None
    return m.group(1).strip().lower() == "true"


def retrieve(retrieval_url: str, query: str, doc_id: str, topk: int = 8) -> List[Dict[str, Any]]:
    r = requests.post(retrieval_url, json={
        "queries": [query],
        "topk": topk,
        "return_scores": True,
        "doc_id": doc_id,
    }, timeout=120)
    r.raise_for_status()
    return r.json()["result"][0]


def passages_to_string(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for i, d in enumerate(docs):
        contents = d["document"]["contents"]
        title = contents.split("\n", 1)[0]
        text = contents.split("\n", 1)[1] if "\n" in contents else ""
        parts.append(f"Doc {i + 1}(Title: {title}) {text}")
    return "\n".join(parts)


_SM_RT_CLIENT = None
_SM_RT_LOCK = threading.Lock()


def _sagemaker_client(region: str):
    global _SM_RT_CLIENT
    if _SM_RT_CLIENT is None:
        with _SM_RT_LOCK:
            if _SM_RT_CLIENT is None:
                import boto3
                from botocore.config import Config
                cfg = Config(max_pool_connections=64, retries={"max_attempts": 3, "mode": "standard"})
                _SM_RT_CLIENT = boto3.Session(profile_name="phase-c", region_name=region).client(
                    "sagemaker-runtime", config=cfg,
                )
    return _SM_RT_CLIENT


def call_searcher(searcher_url: str, system: str, user: str, max_tokens: int = 500) -> str:
    """Searcher invoke — supports either OpenAI HTTP or `sagemaker:<endpoint>` schemes.

    For sagemaker, the URL syntax is `sagemaker:<endpoint_name>` and we read the
    region from SAGEMAKER_REGION env (defaults to ap-northeast-2 where our
    endpoint lives).
    """
    payload = {
        "model": "finqa-qwen25-7b-ppo-s3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if searcher_url.startswith("sagemaker:"):
        endpoint = searcher_url.split(":", 1)[1]
        region = os.getenv("SAGEMAKER_REGION", "ap-northeast-2")
        rt = _sagemaker_client(region)
        resp = rt.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )
        body = json.loads(resp["Body"].read())
        return body["choices"][0]["message"]["content"]
    r = requests.post(searcher_url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_s3_loop(
    question: str,
    doc_id: str,
    searcher_url: str,
    retrieval_url: str,
    topk: int = 8,
    max_turns: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the s3 paper Algorithm 1 multi-turn loop.

    Returns (D_s3, naive_docs):
        D_s3: cumulative <important_info>-selected docs (full text)
        naive_docs: initial top-k from naive RAG (baseline)
    """
    naive_docs = retrieve(retrieval_url, question, doc_id, topk)
    d_s3: List[Dict[str, Any]] = []
    d_s3_keys: set = set()

    def select_important(pool: List[Dict[str, Any]], indices: List[int]):
        for idx in indices:
            i0 = idx - 1
            if 0 <= i0 < len(pool):
                doc = pool[i0]
                title = doc["document"]["contents"].split("\n", 1)[0]
                key = title[:120]
                if key in d_s3_keys:
                    continue
                d_s3_keys.add(key)
                text = doc["document"]["contents"].split("\n", 1)[1] if "\n" in doc["document"]["contents"] else ""
                d_s3.append({
                    "id": f"d_s3_{len(d_s3)}",
                    "text": text,
                    "original_index": title,
                    "source_type": "extracted",
                })

    # Turn 0: model emits decision over naive top-k
    current_pool = naive_docs
    for turn in range(max_turns + 1):
        ctx = passages_to_string(current_pool)
        user = (
            f"<question>{question}</question>\n\n"
            f"<information>{ctx}</information>\n\n"
            "Emit the strict s3 tag sequence."
        )
        out = call_searcher(searcher_url, S3_SYSTEM_PROMPT, user)
        important = parse_important(out)
        complete = parse_complete(out)
        select_important(current_pool, important)
        if complete is True or complete is None:
            break
        next_q = parse_query(out)
        if not next_q:
            break
        try:
            current_pool = retrieve(retrieval_url, next_q, doc_id, topk)
        except Exception:
            break

    return d_s3, naive_docs


_PHASEB_LOADED = False


def score_with_step4(
    question: str,
    doc_id: str,
    chunks: List[Dict[str, Any]],
    gold_answer: Any,
) -> Tuple[bool, str]:
    """Call Step 4 G(Q, chunks) and check numeric match. Returns (is_correct, predicted).

    Routes the Kimi K2.5 call through OpenRouter when OPENROUTER_API_KEY is set,
    otherwise falls back to Bedrock. For exemplar selection we still need an
    embedding — when OpenRouter mode is active and Bedrock embedding is
    unavailable, falls back to fixed first-N exemplars (loses similarity
    relevance but keeps PoT prompt structure).
    """
    global _PHASEB_LOADED
    import sys
    repo_root = Path(os.getenv(
        "FINQA_REPO_ROOT",
        "/Users/mason/project/personal/fin-qa-research-wiki/raw/fin-qa-research",
    ))
    sys.path.insert(0, str(repo_root / "src" / "finqa_common" / "src"))
    sys.path.insert(0, str(repo_root / "src" / "nr-pb-step-4-s3-integrated-policy" / "code"))
    sys.path.insert(0, str(repo_root / "src" / "nr-pb-step-4-s3-integrated-policy" / "train"))
    from finqa_common.utils import (
        execute_program, extract_number, extract_python_code,
    )
    from finqa_common.answer_normalizer import check_answer_match, post_process_answer
    import inference as phaseb_inference

    if not _PHASEB_LOADED:
        try:
            phaseb_inference._load_fsp_state()
        except Exception as e:
            print(f"[score_with_step4] _load_fsp_state warn: {e}")
        _PHASEB_LOADED = True

    if not chunks:
        return False, "(no D_s3)"

    use_openrouter = bool(os.getenv("OPENROUTER_API_KEY", "").strip())

    # Exemplars: try Bedrock embed → similarity, else fall back to first-N from cache.
    exemplars = []
    try:
        from finqa_common.utils import get_bedrock_client, get_embedding
        bedrock = get_bedrock_client(region="us-east-1")
        query_emb = get_embedding(question, "amazon.titan-embed-text-v2:0", bedrock)
        if query_emb:
            exemplars = phaseb_inference.select_exemplars_by_similarity(
                query_embedding=query_emb, shot_number=5, exclude_doc_id=doc_id,
            )
    except Exception:
        pass
    if not exemplars and hasattr(phaseb_inference, "_FSP_CACHE"):
        # Fixed first-N fallback when embedding unavailable (OpenRouter w/o Titan)
        cache = phaseb_inference._FSP_CACHE
        if isinstance(cache, dict):
            exemplars = list(cache.get("examples", []))[:5]

    system_prompt, user_prompt = phaseb_inference.build_fewshot_program_prompt(
        question, chunks, exemplars,
    )

    # Generation: OpenRouter Kimi or Bedrock Kimi
    if use_openrouter:
        import requests as _req
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        model = os.getenv("OPENROUTER_GENERATOR_MODEL", "moonshotai/kimi-k2-0905")
        r = _req.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 2048,
                "temperature": 0.0,
            },
            timeout=180,
        )
        r.raise_for_status()
        generated = r.json()["choices"][0]["message"]["content"]
    else:
        from finqa_common.utils import generate_answer
        generated = generate_answer(
            user_prompt, "moonshotai.kimi-k2.5", bedrock,
            2048, 0.0, system_prompt=system_prompt,
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
    predicted = post_process_answer(predicted, gold_answer)
    is_correct, _ = check_answer_match(predicted, gold_answer)
    return bool(is_correct), str(predicted)


_PRINT_LOCK = threading.Lock()


def _eval_one(q: Dict[str, Any], idx: int, total: int, args) -> Dict[str, Any]:
    question = q["qa"]["question"]
    doc_id = q["id"]
    gold = q["qa"].get("exe_ans", q["qa"].get("answer"))
    try:
        d_s3, naive_docs = run_s3_loop(
            question, doc_id,
            args.searcher_url, args.retrieval_url,
            topk=args.topk, max_turns=args.max_turns,
        )
        naive_chunks = []
        for j, d in enumerate(naive_docs[: args.topk]):
            contents = d["document"]["contents"]
            title = contents.split("\n", 1)[0]
            text = contents.split("\n", 1)[1] if "\n" in contents else ""
            naive_chunks.append({
                "id": f"naive_{j}", "text": text,
                "original_index": title, "source_type": "naive",
            })

        naive_correct, naive_pred = score_with_step4(question, doc_id, naive_chunks, gold)
        s3_correct, s3_pred = score_with_step4(question, doc_id, d_s3 or naive_chunks, gold)
        with _PRINT_LOCK:
            print(f"[{idx}/{total}] {doc_id} | D_s3={len(d_s3)} naive={len(naive_docs)} | "
                  f"naive={naive_correct}({naive_pred}) s3={s3_correct}({s3_pred}) gold={gold}")
        return {
            "id": doc_id, "question": question, "gold": gold,
            "naive_correct": naive_correct, "naive_pred": str(naive_pred),
            "s3_correct": s3_correct, "s3_pred": str(s3_pred),
            "d_s3_chunks": len(d_s3),
        }
    except Exception as e:
        with _PRINT_LOCK:
            print(f"[{idx}/{total}] {doc_id} ERR: {e}")
        return {"id": doc_id, "question": question, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finqa-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--searcher-url", required=True,
                        help="vLLM OpenAI endpoint, e.g. http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:3000/retrieve")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel question workers (ThreadPoolExecutor)")
    parser.add_argument("--out", type=Path, default=Path("/tmp/eval_dev10.json"))
    args = parser.parse_args()

    fname = {"dev": "dev.json", "train": "train.json", "test": "test.json"}[args.split]
    items = json.load(open(args.finqa_dir / fname))[: args.limit]
    total = len(items)

    # Warm up shared state once before parallel execution to avoid race in
    # _PHASEB_LOADED / get_bedrock_client / weaviate connection.
    if items:
        _ = score_with_step4("warmup question", items[0]["id"], [], items[0]["qa"].get("exe_ans"))

    results: List[Dict[str, Any]] = [None] * total  # type: ignore
    t0 = time.time()
    if args.concurrency <= 1:
        for i, q in enumerate(items, 1):
            results[i - 1] = _eval_one(q, i, total, args)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(_eval_one, q, i + 1, total, args): i for i, q in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()

    elapsed = time.time() - t0
    naive_acc = sum(1 for r in results if r.get("naive_correct")) / max(len(results), 1)
    s3_acc = sum(1 for r in results if r.get("s3_correct")) / max(len(results), 1)

    summary = {
        "n": len(results),
        "naive_correct": sum(1 for r in results if r.get("naive_correct")),
        "s3_correct": sum(1 for r in results if r.get("s3_correct")),
        "naive_acc": naive_acc,
        "s3_acc": s3_acc,
        "delta": s3_acc - naive_acc,
        "elapsed_sec": elapsed,
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== summary ===")
    print(f"  n: {len(results)}")
    print(f"  naive acc: {naive_acc:.3f}")
    print(f"  s3 acc:    {s3_acc:.3f}")
    print(f"  delta:     {s3_acc - naive_acc:+.3f}")
    print(f"  elapsed:   {elapsed:.1f}s")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
