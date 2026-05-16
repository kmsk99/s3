"""Convert FinQA train/dev/test JSON → s3 paper parquet format + naive RAG cache.

s3 paper §3.5 / Appendix uses parquet inputs to verl with the following columns:
- prompt           : full user prompt including <question> tags (read by
                     generation_s3.py line 303 via re.findall r"<question>...</question>")
- ground_truth     : the gold-answer field used by the reward function
- data_source      : a tag used by verl's reward dispatcher (e.g., "finqa_exec_acc")
- extra_info       : dict carrying question, golden_answers, doc_id, gold_inds

Naive RAG cache (matches s3 paper line 423):
- Run naive top-k retrieval + the same frozen Step 4 few-shot PoT generator
  used by the PPO reward on every train question.
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
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

S3_ROOT = Path(__file__).resolve().parents[1]
if str(S3_ROOT) not in sys.path:
    sys.path.insert(0, str(S3_ROOT))

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

NAIVE_CACHE_EVALUATOR = "step4_fewshot_pot_v1"
_FINQA_EXEC_ACC_MODULE = None


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


def _fake_solution_for_naive_step4(
    question: str,
    doc_id: str,
    docs: List[Dict[str, Any]],
) -> str:
    """Build a minimal s3 trajectory so reward_score.finqa_exec_acc evaluates
    one-shot D_RAG with the exact same Step 4 generator path as D_s3.

    Retrieval stays naive: one query q_0=Q and the initial top-k docs. The
    <important_info> tag selects all retrieved docs as D_RAG.
    """
    context = passages_to_string(docs)
    selected = ", ".join(str(i) for i in range(1, len(docs) + 1))
    return (
        f'<question>{question}</question>\n'
        f'doc_id: "{doc_id}"\n'
        f"<information>{context}</information>\n"
        f"<important_info>[{selected}]</important_info>\n"
        "<search_complete>True</search_complete>\n"
    )


def _step4_fewshot_correct(
    question: str,
    doc_id: str,
    docs: List[Dict[str, Any]],
    gold: Any,
) -> bool:
    """Acc(Step4-G(Q, D_RAG), A) for the naive baseline term.

    This intentionally imports the VERL reward hook so the naive baseline and
    PPO reward share the same frozen generator/evaluator:
      Acc(G(Q, D_s3), A) - Acc(G(Q, D_RAG), A)
    """
    compute_score_finqa_rag = _load_finqa_exec_acc_module().compute_score_finqa_rag

    solution = _fake_solution_for_naive_step4(question, doc_id, docs)
    score, _, _ = compute_score_finqa_rag(solution, str(gold), data_source="finqa_exec_acc")
    return bool(score >= 1.0)


def _load_finqa_exec_acc_module():
    """Load the FinQA reward file directly without importing the full verl package.

    The reward function itself is the contract we need for Step 4 alignment.
    Importing ``verl.utils...`` also imports training/runtime protocol modules,
    which can require environment-specific dependencies (for example
    tensordict) that are irrelevant to offline dataset preparation.
    """
    global _FINQA_EXEC_ACC_MODULE
    if _FINQA_EXEC_ACC_MODULE is not None:
        return _FINQA_EXEC_ACC_MODULE

    reward_path = S3_ROOT / "verl" / "utils" / "reward_score" / "finqa_exec_acc.py"
    spec = importlib.util.spec_from_file_location("_finqa_exec_acc_direct", reward_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FinQA reward module from {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _FINQA_EXEC_ACC_MODULE = module
    return module


def naive_rag_correct(item: Dict[str, Any], topk: int,
                       retrieval_url: str, generator_url: str,
                       naive_eval_mode: str) -> bool:
    question = item["qa"]["question"]
    doc_id = item["id"]
    gold = item["qa"].get("exe_ans", item["qa"].get("answer"))
    try:
        docs = retrieve_initial(question, doc_id, topk, retrieval_url)
        if naive_eval_mode == "step4":
            return _step4_fewshot_correct(question, doc_id, docs, gold)

        # Legacy smoke mode: direct numeric answer from the generator adapter.
        # Kept only for comparison/debugging; full retraining should use step4.
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


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _cache_meta(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "evaluator": NAIVE_CACHE_EVALUATOR if args.naive_eval_mode == "step4" else "legacy_simple_numeric_v1",
        "naive_eval_mode": args.naive_eval_mode,
        "retrieval": {
            "topk": args.topk,
            "contract": "one-shot q0=question, doc_id-scoped retrieval",
        },
        "generator": {
            "reward_generator": os.getenv("FINQA_REWARD_GENERATOR", "moonshotai.kimi-k2.5"),
            "reward_shots": int(os.getenv("FINQA_REWARD_SHOTS", "5")),
            "reward_temp": float(os.getenv("FINQA_REWARD_TEMP", "0.0")),
            "pipeline": "Step 4 few-shot PoT + execute_program + answer_normalizer"
            if args.naive_eval_mode == "step4"
            else "direct final-number-only prompt + numeric_match",
        },
    }


def _cache_compatible(existing: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    keys = [
        ("evaluator",),
        ("naive_eval_mode",),
        ("retrieval", "topk"),
        ("generator", "reward_generator"),
        ("generator", "reward_shots"),
        ("generator", "reward_temp"),
    ]
    for path in keys:
        lhs: Any = existing
        rhs: Any = expected
        for key in path:
            if not isinstance(lhs, dict) or not isinstance(rhs, dict):
                return False
            lhs = lhs.get(key)
            rhs = rhs.get(key)
        if lhs != rhs:
            return False
    return True


def _write_naive_cache(
    cache_path: Path,
    expected_meta: Dict[str, Any],
    args: argparse.Namespace,
    cache: Dict[str, bool],
    status: str,
) -> None:
    """Persist naive correctness cache and compatible metadata.

    Metadata is written even for in-progress caches so a long Step 4 precompute
    can resume safely after interruption instead of falling back to a stale
    no-metadata cache.
    """
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    meta = dict(expected_meta)
    meta.update({
        "status": status,
        "updated_at_unix": int(time.time()),
        "cache_entries": len(cache),
        "cache_correct": sum(cache.values()),
        "finqa_dir": str(args.finqa_dir),
    })
    if status == "complete":
        meta["completed_at_unix"] = meta["updated_at_unix"]
    with open(_cache_meta_path(cache_path), "w") as f:
        json.dump(meta, f, indent=2)


def build_prompt(item: Dict[str, Any], initial_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Compose chat messages for the s3 search agent.

    Returns list[{role,content}] consumed by tokenizer.apply_chat_template
    (rl_dataset.py expects this exact shape).
    """
    question = item["qa"]["question"]
    context = passages_to_string(initial_docs)
    user_content = (
        f"<question>{question}</question>\n\n"
        f"Initial retrieval (naive RAG top-{len(initial_docs)}):\n"
        f"<information>{context}</information>\n\n"
        "Decide whether the searched results are enough. Emit the strict "
        "tag sequence."
    )
    return [
        {"role": "system", "content": S3_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_parquet_row(
    item: Dict[str, Any],
    split_name: str,
    row_index: int,
    topk: int,
    retrieval_url: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve initial docs and build one s3 parquet row."""
    try:
        initial = retrieve_initial(
            item["qa"]["question"], item["id"], topk, retrieval_url
        )
    except Exception as e:
        print(f"  retrieve err {item['id']}: {e}")
        return None
    prompt = build_prompt(item, initial)
    ground_truth = item["qa"].get("exe_ans", item["qa"].get("answer"))
    gt_str = str(ground_truth)
    return {
        "prompt": prompt,
        "data_source": "finqa_exec_acc",
        "reward_model": {
            "style": "finqa_exec_acc",
            "ground_truth": gt_str,
        },
        "extra_info": {
            "split": split_name,
            "index": row_index,
            "question": item["qa"]["question"],
            "golden_answers": [gt_str],
            "doc_id": item["id"],
        },
    }


def build_rows_parallel(
    items: List[Dict[str, Any]],
    split_name: str,
    topk: int,
    retrieval_url: str,
    workers: int,
) -> List[Dict[str, Any]]:
    """Build parquet rows with bounded retrieval concurrency."""
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    max_pending = max(1, workers * 2)
    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending: Dict[Any, int] = {}
        iterator = iter(enumerate(items))

        def fill_pending() -> None:
            while len(pending) < max_pending:
                try:
                    idx, item = next(iterator)
                except StopIteration:
                    break
                fut = ex.submit(
                    build_parquet_row,
                    item, split_name, idx, topk, retrieval_url,
                )
                pending[fut] = idx

        fill_pending()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                idx = pending.pop(fut)
                row = fut.result()
                done_count += 1
                if row is not None:
                    rows_by_index[idx] = row
                if done_count % 100 == 0:
                    print(
                        f"  {split_name}: {done_count}/{len(items)} initial retrieval done, "
                        f"rows so far: {len(rows_by_index)}"
                    )
            fill_pending()

    return [rows_by_index[i] for i in sorted(rows_by_index)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finqa-dir", type=Path, required=True,
                        help="Directory containing train.json/dev.json/test.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--retrieval-url", type=str, default="http://127.0.0.1:3000/retrieve")
    parser.add_argument("--generator-url", type=str, default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--topk", type=int, default=8, help="Initial naive RAG top-k (s3 paper default 3, scaled for FinQA tables)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--naive-eval-mode", choices=["step4", "simple"], default="step4",
                        help="How to compute naive_correct. step4 matches PPO reward's frozen Step 4 few-shot PoT generator; simple is legacy smoke mode.")
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
    os.environ.setdefault("FINQA_REPO_ROOT", str(args.finqa_dir.resolve().parent))

    splits = {
        "train": args.finqa_dir / "train.json",
        "valid": args.finqa_dir / "dev.json",
        "test": args.finqa_dir / "test.json",
    }
    for name, p in splits.items():
        if not p.exists():
            print(f"  warning: {p} not found, skipping {name}")

    naive_cache_path = args.out_dir / "naive_correct.json"
    naive_meta_path = _cache_meta_path(naive_cache_path)
    expected_meta = _cache_meta(args)
    cache: Dict[str, bool] = {}
    if naive_cache_path.exists():
        existing_meta: Optional[Dict[str, Any]] = None
        if naive_meta_path.exists():
            with open(naive_meta_path) as f:
                existing_meta = json.load(f)
        if existing_meta and _cache_compatible(existing_meta, expected_meta):
            with open(naive_cache_path) as f:
                cache = json.load(f)
            print(
                f"Loaded compatible naive_correct cache: {len(cache)} entries "
                f"({expected_meta['evaluator']})"
            )
        else:
            reason = "missing metadata" if existing_meta is None else "metadata mismatch"
            msg = (
                f"Ignoring stale naive_correct cache ({reason}); "
                f"expected evaluator={expected_meta['evaluator']}, topk={args.topk}."
            )
            if args.skip_precompute:
                raise SystemExit(f"{msg} Cannot --skip-precompute with stale cache.")
            print(msg)

    train_items = load_finqa(splits["train"])
    precompute_items = train_items[: args.limit] if args.limit else train_items

    if not args.skip_precompute:
        print(
            f"Precomputing naive RAG correctness on {len(precompute_items)} train items "
            f"with evaluator={expected_meta['evaluator']}..."
        )
        # Persist an in-progress metadata file immediately. This lets interrupted
        # long runs resume from the partial cache while still rejecting legacy or
        # mismatched caches.
        _write_naive_cache(naive_cache_path, expected_meta, args, cache, "in_progress")
        to_run = [it for it in precompute_items if question_key(it) not in cache]
        print(f"  {len(cache)} cached, {len(to_run)} to run")
        max_pending = max(1, args.workers * 2)
        done_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            pending: Dict[Any, Dict[str, Any]] = {}
            iterator = iter(to_run)

            def fill_pending() -> None:
                while len(pending) < max_pending:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    fut = ex.submit(
                        naive_rag_correct, item, args.topk,
                        args.retrieval_url, args.generator_url,
                        args.naive_eval_mode,
                    )
                    pending[fut] = item

            fill_pending()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    it = pending.pop(fut)
                    cache[question_key(it)] = bool(fut.result())
                    done_count += 1
                    if done_count % 50 == 0:
                        _write_naive_cache(naive_cache_path, expected_meta, args, cache, "in_progress")
                        print(f"  {done_count}/{len(to_run)} naive precompute done, "
                              f"correct so far: {sum(cache.values())}")
                fill_pending()
        _write_naive_cache(naive_cache_path, expected_meta, args, cache, "complete")
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
        candidates = []
        for it in items:
            if split_name == "train" and args.hard_only:
                k = question_key(it)
                if cache.get(k, False):
                    continue
            candidates.append(it)
        print(f"Building {split_name} parquet rows from {len(candidates)} candidates with {args.workers} workers...")
        rows = build_rows_parallel(
            candidates, split_name, args.topk, args.retrieval_url, args.workers
        )
        if not rows:
            print(f"  no rows for {split_name}")
            continue
        out_path = args.out_dir / f"{split_name}_finqa_s3.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path)
        print(f"  wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
