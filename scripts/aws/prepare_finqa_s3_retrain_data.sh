#!/bin/bash
# Prepare full-size FinQA parquet inputs for s3 PPO retraining.
#
# Safety:
# - Never passes --force.
# - Never overwrites the default s3://$S3_BUCKET/finqa_s3/ prefix by default.
# - Uploads only when --upload is explicitly provided, to a timestamped prefix
#   unless --s3-prefix is specified.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S3_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WIKI_ROOT=$(cd "$S3_ROOT/.." && pwd)

OUT_DIR="$S3_ROOT/data/finqa_s3_full"
UPLOAD=0
S3_PREFIX=""
PROFILE="${AWS_PROFILE:-phase-c}"
AWS_REGION="${AWS_REGION:-us-east-1}"
WORKERS="${WORKERS:-8}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:3000/retrieve}"
GENERATOR_URL="${GENERATOR_LLM_URL:-${GENERATOR_URL:-http://127.0.0.1:8000/v1/chat/completions}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --upload) UPLOAD=1; shift ;;
    --s3-prefix) S3_PREFIX="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --region) AWS_REGION="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
    --generator-url) GENERATOR_URL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --force)
      echo "ERROR: --force is forbidden for this repository's FinQA/s3 retraining flow." >&2
      exit 2
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

# Best-effort env loading for S3_BUCKET only; secrets are not printed.
if [[ -f "$WIKI_ROOT/.env.aws.phase-c.runtime" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$WIKI_ROOT/.env.aws.phase-c.runtime"; set +a
fi

CACHE_PATH="$OUT_DIR/naive_correct.json"
if [[ ! -f "$CACHE_PATH" && -n "${S3_BUCKET:-}" ]]; then
  echo "Downloading existing naive_correct cache candidate from s3://$S3_BUCKET/finqa_s3/ ..."
  aws s3 cp "s3://$S3_BUCKET/finqa_s3/naive_correct.json" "$CACHE_PATH" \
    --profile "$PROFILE" --region "$AWS_REGION" || true
  aws s3 cp "s3://$S3_BUCKET/finqa_s3/naive_correct.json.meta.json" "$CACHE_PATH.meta.json" \
    --profile "$PROFILE" --region "$AWS_REGION" || true
fi

TRAIN_COUNT=$($PYTHON_BIN - <<PY
import json
from pathlib import Path
p=Path("$WIKI_ROOT/raw/fin-qa-research/dataset/train.json")
print(len(json.loads(p.read_text())))
PY
)
echo "prepare_finqa_dataset.py will reuse only metadata-compatible Step 4 few-shot cache entries."
echo "If the existing cache is legacy/simple or incomplete, it will recompute missing entries."
echo "This may call the generator and incur external model cost."

echo "Building full FinQA s3 parquet without --limit ..."
(
  export FINQA_REPO_ROOT="$WIKI_ROOT/raw/fin-qa-research"
  cd "$S3_ROOT"
  "$PYTHON_BIN" integrations/prepare_finqa_dataset.py \
    --finqa-dir "$WIKI_ROOT/raw/fin-qa-research/dataset" \
    --out-dir "$OUT_DIR" \
    --retrieval-url "$RETRIEVAL_URL" \
    --generator-url "$GENERATOR_URL" \
    --topk 8 \
    --workers "$WORKERS" \
    --naive-eval-mode step4 \
    --splits train,valid,test
)

"$PYTHON_BIN" "$S3_ROOT/integrations/inspect_finqa_s3_parquet.py" "$OUT_DIR" \
  --expected-train-min 1000 \
  --expected-valid-min 800 \
  --expected-test-min 1000 \
  --expected-naive-cache "$TRAIN_COUNT"

if [[ "$UPLOAD" -eq 1 ]]; then
  if [[ -z "${S3_BUCKET:-}" && -z "$S3_PREFIX" ]]; then
    echo "ERROR: S3_BUCKET not set; pass --s3-prefix s3://bucket/prefix/ or source runtime env." >&2
    exit 2
  fi
  if [[ -z "$S3_PREFIX" ]]; then
    TS=$(date +%Y%m%d-%H%M%S)
    S3_PREFIX="s3://$S3_BUCKET/finqa_s3_full_$TS/"
  fi
  if [[ "$S3_PREFIX" == "s3://$S3_BUCKET/finqa_s3/" || "$S3_PREFIX" == "s3://$S3_BUCKET/finqa_s3" ]]; then
    echo "ERROR: refusing to overwrite default finqa_s3 prefix. Use a versioned prefix first." >&2
    exit 2
  fi
  echo "Uploading prepared parquet to $S3_PREFIX ..."
  aws s3 sync "$OUT_DIR/" "$S3_PREFIX" --profile "$PROFILE" --region "$AWS_REGION"
  REL_PREFIX="${S3_PREFIX#s3://$S3_BUCKET/}"
  REL_PREFIX="${REL_PREFIX%/}"
  echo "Uploaded. Launch with: FINQA_S3_PREFIX=$REL_PREFIX bash s3/scripts/aws/aws_launch_s3_paper_ec2.sh full"
fi
