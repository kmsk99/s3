"""Re-run only the failed items from a previous eval and merge back."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
import eval_dev10 as ev

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--finqa-dir", type=Path, required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--searcher-url", required=True)
    p.add_argument("--retrieval-url", default="http://127.0.0.1:3000/retrieve")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    prev = json.load(open(args.in_json))
    fname = {"dev": "dev.json", "train": "train.json", "test": "test.json"}[args.split]
    items_all = json.load(open(args.finqa_dir / fname))
    by_id = {q["id"]: q for q in items_all}

    failed = [r for r in prev["results"] if "error" in r]
    print(f"Re-running {len(failed)} failed items")
    redo = [by_id[r["id"]] for r in failed if r["id"] in by_id]

    # Warmup
    if redo:
        ev.score_with_step4("warmup", redo[0]["id"], [], redo[0]["qa"].get("exe_ans"))

    fixed: dict = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(ev._eval_one, q, i + 1, len(redo), args): q["id"]
                for i, q in enumerate(redo)}
        for fut in as_completed(futs):
            qid = futs[fut]
            fixed[qid] = fut.result()

    new_results = []
    for r in prev["results"]:
        if "error" in r and r["id"] in fixed:
            new_results.append(fixed[r["id"]])
        else:
            new_results.append(r)

    naive_correct = sum(1 for r in new_results if r.get("naive_correct"))
    s3_correct = sum(1 for r in new_results if r.get("s3_correct"))
    n = len(new_results)
    naive_acc = naive_correct / n
    s3_acc = s3_correct / n

    summary = {
        "n": n, "naive_correct": naive_correct, "s3_correct": s3_correct,
        "naive_acc": naive_acc, "s3_acc": s3_acc, "delta": s3_acc - naive_acc,
        "elapsed_sec": prev.get("elapsed_sec"),
        "results": new_results,
    }
    args.out_json.write_text(json.dumps(summary, indent=2, default=str))
    still_err = sum(1 for r in new_results if "error" in r)
    print(f"\n=== merged ===")
    print(f"  n={n}  scored={n - still_err}  errors={still_err}")
    print(f"  naive: {naive_correct}/{n} = {naive_acc:.4f}")
    print(f"  s3:    {s3_correct}/{n} = {s3_acc:.4f}")
    print(f"  Δ:     {s3_acc - naive_acc:+.4f}")
    print(f"  → {args.out_json}")

if __name__ == "__main__":
    main()
