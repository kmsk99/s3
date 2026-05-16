#!/bin/bash
# Launch p4d.24xlarge for s3 paper PPO training on FinQA.
#
# v2: Removed conda dependency (DLAMI 2026 has no conda in PATH for ubuntu user).
#     Uses uv to create separate Python 3.10 venv for VERL.
#     Restores Weaviate from S3 snapshot (no in-EC2 ingest).
#     Pulls parquet from S3 (no in-EC2 generation).
#
# Single instance hosts:
#   - Weaviate (Docker, port 8080, restored from snapshot)
#   - finqa_retrieval_adapter (FastAPI, port 3000)
#   - finqa_generator_adapter (FastAPI, port 8000 → Bedrock Kimi K2.5)
#   - VERL PPO training on all 8 GPUs

set -euo pipefail

WIKI_ROOT="/Users/mason/project/personal/fin-qa-research-wiki"
ENV_FILE="$WIKI_ROOT/.env.aws.phase-c"
RUNTIME_FILE="$WIKI_ROOT/.env.aws.phase-c.runtime"

if [ ! -f "$ENV_FILE" ] || [ ! -f "$RUNTIME_FILE" ]; then
    echo "ERROR: missing env files; run aws_provision.sh first"
    exit 1
fi

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; source "$RUNTIME_FILE"; set +a
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${AWS_SECRET:-}}"
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"

PROFILE="phase-c"
MODE="${1:-smoke}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p4d.24xlarge}"
FINQA_S3_PREFIX="${FINQA_S3_PREFIX:-finqa_s3}"
CODE_TARBALL_KEY="${CODE_TARBALL_KEY:-s3-paper-code-launch-$(date -u +%Y%m%dT%H%M%SZ).tar.gz}"
SUBNET_ID="${SUBNET_ID:-}"

case "$MODE" in
    smoke) TOTAL_STEPS=1 ; INSTANCE_LIFETIME="1 hour" ;;
    poc)   TOTAL_STEPS=5 ; INSTANCE_LIFETIME="3 hours" ;;
    full)  TOTAL_STEPS=20 ; INSTANCE_LIFETIME="8 hours" ;;
    *) echo "ERROR: mode must be smoke|poc|full"; exit 1 ;;
esac

PURCHASE="${PURCHASE:-}"
if [ -z "$PURCHASE" ]; then
    case "$MODE" in
        smoke|poc) PURCHASE="on-demand" ;;
        full)      PURCHASE="spot" ;;
    esac
fi
echo "Mode: $MODE ($INSTANCE_LIFETIME budget) Purchase: $PURCHASE Instance: $INSTANCE_TYPE"
echo "TOTAL_STEPS: $TOTAL_STEPS (s3 paper §3.5 uses 20 steps for full run)"
echo "FINQA_S3_PREFIX: s3://$S3_BUCKET/$FINQA_S3_PREFIX/"
echo "CODE_TARBALL_KEY: s3://$S3_BUCKET/$CODE_TARBALL_KEY"
if [ -n "$SUBNET_ID" ]; then
    echo "SUBNET_ID: $SUBNET_ID"
fi

AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04) *" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text --profile "$PROFILE" --region us-east-1)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "ERROR: could not resolve Deep Learning AMI"
    exit 1
fi
echo "AMI: $AMI_ID"

CODE_TARBALL=$(mktemp)
echo "Packaging current s3/ code for EC2 bootstrap..."
tar \
    --exclude='.git' \
    --exclude='.omc' \
    --exclude='.omx' \
    --exclude='__pycache__' \
    --exclude='data' \
    --exclude='train_logs' \
    --exclude='verl_checkpoints' \
    -czf "$CODE_TARBALL" \
    -C "$WIKI_ROOT/s3" .
aws s3 cp "$CODE_TARBALL" "s3://$S3_BUCKET/$CODE_TARBALL_KEY" \
    --profile "$PROFILE" --region us-east-1
echo "Uploaded code tarball."

# Defaults are expanded into the bootstrap script below. Keep them defined in
# the launcher shell as well, because this script runs with `set -u` and the
# bootstrap heredoc intentionally expands local AWS/data variables.
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-1000}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-800}"
MIN_TEST_ROWS="${MIN_TEST_ROWS:-1000}"

BOOTSTRAP=$(mktemp)
cat > "$BOOTSTRAP" <<EOF
#!/bin/bash
exec > /var/log/s3-paper-bootstrap.log 2>&1
set -eo pipefail

echo "[bootstrap] start: \$(date)"
cd /home/ubuntu

# 1. Code tarballs from S3 (pitfall #12 — private repo workaround)
sudo -u ubuntu bash -c "mkdir -p /home/ubuntu/fin-qa-research /home/ubuntu/s3"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/fin-qa-code.tar.gz /home/ubuntu/fin-qa-code.tar.gz"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/$CODE_TARBALL_KEY /home/ubuntu/s3-paper-code.tar.gz"
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && tar -xzf /home/ubuntu/fin-qa-code.tar.gz"
sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && tar -xzf /home/ubuntu/s3-paper-code.tar.gz"
sudo -u ubuntu bash -c "mkdir -p /home/ubuntu/fin-qa-research/dataset"
echo "[bootstrap] code extracted: \$(date)"

# 2. Install uv (pitfall #8 — uv handles venv paths cleanly)
sudo -u ubuntu bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"

# 3. FinQA venv (Python 3.12, fin-qa-research/pyproject.toml)
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && /home/ubuntu/.local/bin/uv sync --no-dev"
# Add adapter deps (fastapi + uvicorn[standard] + pyarrow)
sudo -u ubuntu bash -c "/home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/fin-qa-research/.venv/bin/python fastapi 'uvicorn[standard]' pyarrow"
echo "[bootstrap] finqa venv ready: \$(date)"

# 4. VERL venv (Python 3.10, s3 paper deps — NO CONDA)
sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && /home/ubuntu/.local/bin/uv venv --python 3.10 .venv-verl"
sudo -u ubuntu bash -c "/home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/s3/.venv-verl/bin/python --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0"
# Build dependencies must be installed BEFORE --no-build-isolation packages (e.g., flash-attn)
sudo -u ubuntu bash -c "/home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/s3/.venv-verl/bin/python setuptools wheel packaging ninja"
sudo -u ubuntu bash -c "/home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/s3/.venv-verl/bin/python 'vllm==0.6.3' 'transformers<4.48' accelerate 'tensordict<0.6' hydra-core ray wandb codetiming datasets dill pybind11 IPython matplotlib pandas numpy pyarrow"
# flash-attn is OPTIONAL — vLLM uses VLLM_ATTENTION_BACKEND=XFORMERS, HF actor/critic use sdpa.
# If build fails (common on DLAMI without matching cuDNN), skip and let HF fall back.
export CUDA_HOME=/usr/local/cuda
sudo -u ubuntu bash -c "CUDA_HOME=/usr/local/cuda /home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/s3/.venv-verl/bin/python --no-build-isolation flash-attn" || echo "[bootstrap] flash-attn skipped (using sdpa/xformers fallback)"
# Install s3 paper as editable package (gives access to verl + s3 modules)
sudo -u ubuntu bash -c "/home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/s3/.venv-verl/bin/python -e /home/ubuntu/s3"
echo "[bootstrap] verl venv ready: \$(date)"

# 5. Weaviate via docker compose v2 (pitfall #6)
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && docker compose up -d weaviate-finqa"
sleep 15

# 6. Restore Weaviate volume from snapshot (skips ~25 min ingest)
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/weaviate-snapshot.tar.gz /home/ubuntu/weaviate-snapshot.tar.gz"
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && docker compose stop weaviate-finqa"
# Extract into the named volume
docker run --rm -v fin-qa-research_weaviate_data_none:/data -v /home/ubuntu:/backup alpine sh -c "rm -rf /data/* && tar -xzf /backup/weaviate-snapshot.tar.gz -C /data"
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && docker compose up -d weaviate-finqa"
sleep 20
# Verify collection present
curl -fsS http://127.0.0.1:8080/v1/schema | python3 -c "import sys,json; cs=json.loads(sys.stdin.read()).get('classes',[]); ok=any(c['class']=='FinQA_Chunking_Markdown' for c in cs); print(f'collections: {[c[\"class\"] for c in cs]}'); exit(0 if ok else 1)"
echo "[bootstrap] weaviate restored: \$(date)"

# 7. Pull pre-generated parquet + naive_correct cache + FinQA dataset from S3
sudo -u ubuntu bash -c "mkdir -p /home/ubuntu/s3/data/finqa_s3 && aws s3 sync s3://$S3_BUCKET/$FINQA_S3_PREFIX/ /home/ubuntu/s3/data/finqa_s3/"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/dataset/train.json /home/ubuntu/fin-qa-research/dataset/train.json"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/dataset/dev.json /home/ubuntu/fin-qa-research/dataset/dev.json"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/dataset/test.json /home/ubuntu/fin-qa-research/dataset/test.json"
echo "[bootstrap] data ready: \$(date)"

# 7b. Preflight parquet row counts. This prevents accidentally reusing a
# smoke/limited parquet set (e.g. train=171 rows) for a full PPO run.
MIN_TRAIN_ROWS=${MIN_TRAIN_ROWS:-1000}
MIN_VALID_ROWS=${MIN_VALID_ROWS:-800}
MIN_TEST_ROWS=${MIN_TEST_ROWS:-1000}
sudo -u ubuntu bash -c "/home/ubuntu/s3/.venv-verl/bin/python /home/ubuntu/s3/integrations/inspect_finqa_s3_parquet.py /home/ubuntu/s3/data/finqa_s3 --expected-train-min $MIN_TRAIN_ROWS --expected-valid-min $MIN_VALID_ROWS --expected-test-min $MIN_TEST_ROWS --expected-naive-cache 6251"
echo "[bootstrap] finqa_s3 parquet preflight passed: \$(date)"

# 8. Spot interrupt handler (pitfall #10 — HTTP 200 only)
cat > /home/ubuntu/spot-watch.sh <<'WATCH'
#!/bin/bash
while true; do
    status=\$(curl -s -o /dev/null -w "%{http_code}" http://169.254.169.254/latest/meta-data/spot/termination-time)
    if [ "\$status" = "200" ]; then
        echo "[spot-watch] TERMINATION imminent — SIGTERM training"
        pkill -TERM -f main_ppo || true
        sleep 90
        aws s3 sync /home/ubuntu/s3/verl_checkpoints/ s3://$S3_BUCKET/verl-checkpoints-emergency/
        break
    fi
    sleep 5
done
WATCH
chmod +x /home/ubuntu/spot-watch.sh
if [ "$PURCHASE" = "spot" ]; then
    sudo -u ubuntu bash -c "/home/ubuntu/spot-watch.sh &"
fi

# 9. Start FinQA retrieval adapter on port 3000 (uses finqa venv)
sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && WEAVIATE_URL=http://127.0.0.1:8080 FINQA_REPO_ROOT=/home/ubuntu/fin-qa-research AWS_REGION=us-east-1 nohup /home/ubuntu/fin-qa-research/.venv/bin/python -m uvicorn integrations.finqa_retrieval_adapter:app --host 0.0.0.0 --port 3000 --workers 4 > /home/ubuntu/retrieval-adapter.log 2>&1 &"

# 10. Start FinQA generator adapter on port 8000 (uses finqa venv)
sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && AWS_REGION=us-east-1 FINQA_REPO_ROOT=/home/ubuntu/fin-qa-research nohup /home/ubuntu/fin-qa-research/.venv/bin/python -m uvicorn integrations.finqa_generator_adapter:app --host 0.0.0.0 --port 8000 --workers 4 > /home/ubuntu/generator-adapter.log 2>&1 &"

sleep 20

# 11. Health checks
curl -fsS http://127.0.0.1:3000/health || (echo "retrieval adapter down"; cat /home/ubuntu/retrieval-adapter.log; exit 1)
curl -fsS http://127.0.0.1:8000/health || (echo "generator adapter down"; cat /home/ubuntu/generator-adapter.log; exit 1)
echo "[bootstrap] adapters healthy: \$(date)"

# 12. Launch VERL PPO training (verl venv, all 8 GPUs)
sudo -u ubuntu bash -c "source /home/ubuntu/s3/.venv-verl/bin/activate && cd /home/ubuntu/s3 && VERL_GPUS=0,1,2,3,4,5,6,7 TOTAL_STEPS=$TOTAL_STEPS GENERATOR_LLM_URL=http://127.0.0.1:8000/v1/chat/completions RETRIEVER_URL=http://127.0.0.1:3000/retrieve bash scripts/train/train_s3_finqa.sh 2>&1 | tee /home/ubuntu/s3-train.log"

echo "[bootstrap] training finished: \$(date)"

# 13. Final S3 sync
sudo -u ubuntu bash -c "aws s3 sync /home/ubuntu/s3/verl_checkpoints/ s3://$S3_BUCKET/verl-checkpoints-final/"

echo "[bootstrap] complete"
EOF

echo "Launching $INSTANCE_TYPE ($PURCHASE) instance..."

MARKET_OPTS=()
if [ "$PURCHASE" = "spot" ]; then
    MARKET_OPTS=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}')
fi
SUBNET_OPTS=()
if [ -n "$SUBNET_ID" ]; then
    SUBNET_OPTS=(--subnet-id "$SUBNET_ID")
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    ${MARKET_OPTS[@]+"${MARKET_OPTS[@]}"} \
    ${SUBNET_OPTS[@]+"${SUBNET_OPTS[@]}"} \
    --key-name "$SSH_KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=phase-c-ec2-role" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=1000,VolumeType=gp3,DeleteOnTermination=true}' \
    --user-data "file://$BOOTSTRAP" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=phase-c-s3-paper},{Key=owner,Value=mason},{Key=mode,Value=$MODE},{Key=purchase,Value=$PURCHASE}]" \
    --query 'Instances[0].InstanceId' \
    --output text --profile "$PROFILE" --region us-east-1)

echo "Instance: $INSTANCE_ID"
rm -f "$BOOTSTRAP"

echo "Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --profile "$PROFILE" --region us-east-1

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text --profile "$PROFILE" --region us-east-1)

echo
echo "============================================================"
echo "Instance ready"
echo "============================================================"
echo "  ID:        $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  SSH:       ssh -i $SSH_KEY_PATH ubuntu@$PUBLIC_IP"
echo "  Bootstrap: tail -f /var/log/s3-paper-bootstrap.log"
echo "  Training:  tail -f /home/ubuntu/s3-train.log"
echo
echo "  Mode: $MODE  Time budget: $INSTANCE_LIFETIME  Steps: $TOTAL_STEPS"
echo "  FinQA data: s3://$S3_BUCKET/$FINQA_S3_PREFIX/"
echo "  Code:      s3://$S3_BUCKET/$CODE_TARBALL_KEY"
echo
echo "INSTANCE_ID=$INSTANCE_ID" >> "$RUNTIME_FILE"
echo "INSTANCE_IP=$PUBLIC_IP" >> "$RUNTIME_FILE"
echo "MODE=$MODE" >> "$RUNTIME_FILE"
echo
echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE"
rm -f "$CODE_TARBALL"
