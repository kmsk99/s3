"""Convert FinQA train/dev/test JSON → s3 paper parquet format + naive RAG cache.

s3 paper §3.5 / Appendix uses parquet inputs to verl with the following columns:
- prompt           : full user prompt including <question> tags (read by
                     generation_s3.py line 303 via re.findall r"<question>...</question>")
- ground_truth     : the gold-answer field used by the reward function
- data_source      : a tag used by verl's reward dispatcher (e.g., "finqa_exec_acc")
- extra_info       : dict carrying question, golden_answers, doc_id, gold_inds

Naive RAG cache (matches s3 paper line 423):
- Run naive top-k retrieval + frozen generator on every train question.
- Save cache mapping question_key → bool (naive_correct).
- Filter to hard-only (naive_correct=False) before writing parquet.

Usage:
    python prepare_finqa_dataset.py \\
        --finqa-dir dataset \\
        --out-dir data/finqa_s3 \\
        --retrieval-url http://127.0.0.1:3000/retrieve \\
        --generator-url http://127.0.0.1:8000/v1/chat/completions \\
        --topk 8 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


# s3 paper system prompt (Appendix B Figure 11, verbatim semantics).
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
    "  (after <information>...</information> arrives)\n"
    "  <important_info>[doc_index, ...]</important_info>\n"
    "  <search_complete>True|False</search_complete>\n"
)


def question_key(item: Dict[str, Any]) -> str:
    """Stable per-question key (matches existing s3-integrated-policy convention)."""
    q = item["qa"]["question"]
    h = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
    return f"{item['id']}::{h}"


def load_finqa(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def retrieve_initial(question: str, doc_id: str, topk: int, url: str) -> List[Dict[str, Any]]:
    """Naive RAG (Begin-with-Search) — top-k for q_0 = Q."""
    r = requests.post(url, json={
        "queries": [question],
        "topk": topk,
        "return_scores": True,
        "doc_id": doc_id,
    }, timeout=120)
    r.raise_for_status()
    return r.json()["result"][0]


def call_generator(messages: List[Dict[str, str]], url: str, max_tokens: int = 256) -> str:
    r = requests.post(url, json={
        "model": "moonshotai.kimi-k2.5",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def passages_to_string(docs: List[Dict[str, Any]]) -> str:
    out = []
    for i, d in enumerate(docs):
        contents = d["document"]["contents"]
        title = contents.split("\n", 1)[0]
        text = contents.split("\n", 1)[1] if "\n" in contents else ""
        out.append(f"Doc {i + 1}(Title: {title}) {text}")
    return "\n".join(out)


def numeric_match(prediction: str, gold: Any) -> bool:
    """FinQA exec_acc semantics: extract first number, compare to gold within 1e-3."""
    if prediction is None:
        return False
    nums = re.findall(r"-?\d+(?:\.\d+)?", prediction.replace(",", ""))
    if not nums:
        return False
    try:
        pred = float(nums[0])
        try:
            g = float(str(gold).replace(",", "").replace("%", ""))
        except ValueError:
            return False
        return abs(pred - g) <= max(1e-3, 1e-3 * abs(g))
    except (ValueError, TypeError):
        return False


def naive_rag_correct(item: Dict[str, Any], topk: int,
                       retrieval_url: str, generator_url: str) -> bool:
    question = item["qa"]["question"]
    doc_id = item["id"]
    gold = item["qa"].get("exe_ans", item["qa"].get("answer"))
    try:
        docs = retrieve_initial(question, doc_id, topk, retrieval_url)
        context = passages_to_string(docs)
        prompt = (
            f"Use the following contexts (some might be irrelevant) on demand:\n\n"
            f"Contexts:\n{context}\n\nQuestion: {question}\n\n"
            "Important: directly answer with the final number only."
        )
        out = call_generator([
            {"role": "system", "content": "You are a precise financial QA assistant."},
            {"role": "user", "content": prompt},
        ], generator_url, max_tokens=128)
        return numeric_match(out, gold)
    except Exception as e:
        print(f"  naive_rag err {doc_id}: {e}")
        return False


def build_prompt(item: Dict[str, Any], initial_docs: List[Dict[str, Any]]) -> str:
    """Compose the prompt the s3 search agent sees at turn 1.

    Includes the system prompt + initial naive RAG retrieval + the question
    wrapped in <question> tags so generation_s3.py can recover it.
    """
    question = item["qa"]["question"]
    context = passages_to_string(initial_docs)
    user = (
        f"<question>{question}</question>\n\n"
        f"Initial retrieval (naive RAG top-{len(initial_docs)}):\n"
        f"<information>{context}</information>\n\n"
        "Decide whether the searched results are enough. Emit the strict "
        "tag sequence."
    )
    return f"<|im_start|>system\n{S3_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finqa-dir", type=Path, required=True,
                        help="Directory containing train.json/dev.json/test.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--retrieval-url", type=str, default="http://127.0.0.1:3000/retrieve")
    parser.add_argument("--generator-url", type=str, default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--topk", type=int, default=8, help="Initial naive RAG top-k (s3 paper default 3, scaled for FinQA tables)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hard-only", action="store_true", default=True,
                        help="Restrict to questions where naive RAG fails (s3 paper line 423)")
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip naive-RAG precompute, assume cache exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap each split at N items (smoke testing)")
    parser.add_argument("--splits", type=str, default="train,valid,test",
                        help="Comma-separated splits to emit (subset of train,valid,test)")
    args = parser.parse_args()

    if not HAVE_PYARROW:
        raise SystemExit("pyarrow required: pip install pyarrow")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": args.finqa_dir / "train.json",
        "valid": args.finqa_dir / "dev.json",
        "test": args.finqa_dir / "test.json",
    }
    for name, p in splits.items():
        if not p.exists():
            print(f"  warning: {p} not found, skipping {name}")

    naive_cache_path = args.out_dir / "naive_correct.json"
    cache: Dict[str, bool] = {}
    if naive_cache_path.exists():
        with open(naive_cache_path) as f:
            cache = json.load(f)

    train_items = load_finqa(splits["train"])

    if not args.skip_precompute:
        print(f"Precomputing naive RAG correctness on {len(train_items)} train items...")
        to_run = [it for it in train_items if question_key(it) not in cache]
        print(f"  {len(cache)} cached, {len(to_run)} to run")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(naive_rag_correct, it, args.topk,
                          args.retrieval_url, args.generator_url): it
                for it in to_run
            }
            for i, fut in enumerate(as_completed(futures), 1):
                it = futures[fut]
                cache[question_key(it)] = bool(fut.result())
                if i % 50 == 0:
                    with open(naive_cache_path, "w") as f:
                        json.dump(cache, f)
                    print(f"  {i}/{len(to_run)} naive precompute done, "
                          f"correct so far: {sum(cache.values())}")
        with open(naive_cache_path, "w") as f:
            json.dump(cache, f)
    print(f"naive_correct: {sum(cache.values())}/{len(cache)} = "
          f"{sum(cache.values())/max(len(cache),1)*100:.1f}%")

    # Build parquet for each split.
    selected_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    for split_name, p in splits.items():
        if split_name not in selected_splits:
            continue
        if not p.exists():
            continue
        items = load_finqa(p)
        if args.limit:
            items = items[: args.limit]
        rows = []
        for it in items:
            if split_name == "train" and args.hard_only:
                k = question_key(it)
                if cache.get(k, False):
                    continue
            try:
                initial = retrieve_initial(
                    it["qa"]["question"], it["id"], args.topk, args.retrieval_url
                )
            except Exception as e:
                print(f"  retrieve err {it['id']}: {e}")
                continue
            prompt = build_prompt(it, initial)
            ground_truth = it["qa"].get("exe_ans", it["qa"].get("answer"))
            rows.append({
                "prompt": prompt,
                "ground_truth": str(ground_truth),
                "data_source": "finqa_exec_acc",
                "extra_info": json.dumps({
                    "question": it["qa"]["question"],
                    "golden_answers": [str(ground_truth)],
                    "doc_id": it["id"],
                    "gold_inds": it["qa"].get("gold_inds", {}),
                }),
            })
        if not rows:
            print(f"  no rows for {split_name}")
            continue
        out_path = args.out_dir / f"{split_name}_finqa_s3.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path)
        print(f"  wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
