#!/usr/bin/env python3
"""Inspect FinQA parquet files prepared for s3 PPO training.

This is a preflight guard for retraining. It fails fast when a smoke/limited
parquet set (for example train=171 rows) is accidentally reused for a full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _load_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pyarrow is required to inspect parquet row counts. "
            "Install pyarrow or run inside the s3/VERL environment. "
            f"Original error: {exc}"
        )
    return pq


def parquet_rows(path: Path) -> int:
    pq = _load_pyarrow()
    return int(pq.ParquetFile(path).metadata.num_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="Directory containing *_finqa_s3.parquet")
    ap.add_argument("--expected-train-min", type=int, default=1000)
    ap.add_argument("--expected-valid-min", type=int, default=800)
    ap.add_argument("--expected-test-min", type=int, default=1000)
    ap.add_argument("--expected-naive-cache", type=int, default=6251)
    ap.add_argument("--expected-naive-evaluator", type=str, default="step4_fewshot_pot_v1")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args()

    files = {
        "train": args.data_dir / "train_finqa_s3.parquet",
        "valid": args.data_dir / "valid_finqa_s3.parquet",
        "test": args.data_dir / "test_finqa_s3.parquet",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        payload = {"ok": False, "error": "missing parquet files", "missing": missing}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload, file=sys.stderr)
        return 2

    rows: Dict[str, int] = {name: parquet_rows(path) for name, path in files.items()}

    cache_path = args.data_dir / "naive_correct.json"
    naive_cache_entries = None
    naive_cache_correct = None
    if cache_path.exists():
        cache: Dict[str, Any] = json.loads(cache_path.read_text())
        naive_cache_entries = len(cache)
        naive_cache_correct = sum(1 for v in cache.values() if bool(v))
    meta_path = Path(str(cache_path) + ".meta.json")
    naive_cache_meta = None
    naive_cache_evaluator = None
    if meta_path.exists():
        naive_cache_meta = json.loads(meta_path.read_text())
        naive_cache_evaluator = naive_cache_meta.get("evaluator")

    checks = {
        "train_min": rows["train"] >= args.expected_train_min,
        "valid_min": rows["valid"] >= args.expected_valid_min,
        "test_min": rows["test"] >= args.expected_test_min,
        "naive_cache_entries": (
            naive_cache_entries is None or naive_cache_entries >= args.expected_naive_cache
        ),
        "naive_cache_evaluator": naive_cache_evaluator == args.expected_naive_evaluator,
    }
    ok = all(checks.values())
    payload = {
        "ok": ok,
        "data_dir": str(args.data_dir),
        "rows": rows,
        "expected_min": {
            "train": args.expected_train_min,
            "valid": args.expected_valid_min,
            "test": args.expected_test_min,
        },
        "naive_cache": {
            "path": str(cache_path),
            "exists": cache_path.exists(),
            "entries": naive_cache_entries,
            "correct": naive_cache_correct,
            "hard": None if naive_cache_entries is None else naive_cache_entries - int(naive_cache_correct or 0),
            "expected_entries_min": args.expected_naive_cache,
            "meta_path": str(meta_path),
            "meta_exists": meta_path.exists(),
            "evaluator": naive_cache_evaluator,
            "expected_evaluator": args.expected_naive_evaluator,
        },
        "checks": checks,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"FinQA s3 parquet: {args.data_dir}")
        print(f"  train rows: {rows['train']} (min {args.expected_train_min})")
        print(f"  valid rows: {rows['valid']} (min {args.expected_valid_min})")
        print(f"  test  rows: {rows['test']} (min {args.expected_test_min})")
        if naive_cache_entries is not None:
            hard = naive_cache_entries - int(naive_cache_correct or 0)
            print(
                "  naive_correct: "
                f"entries={naive_cache_entries}, correct={naive_cache_correct}, hard={hard} "
                f"(entries min {args.expected_naive_cache})"
            )
            print(
                "  naive evaluator: "
                f"{naive_cache_evaluator or 'MISSING'} "
                f"(expected {args.expected_naive_evaluator})"
            )
        print("  status:", "OK" if ok else "FAIL")

    if not ok:
        print(
            "ERROR: FinQA s3 data looks like a smoke/limited build; "
            "regenerate without --limit before full PPO training.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
