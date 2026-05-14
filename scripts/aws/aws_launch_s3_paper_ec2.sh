#!/bin/bash
# Launch p4d.24xlarge for s3 paper PPO training on FinQA.
#
# Single instance hosts:
#   - Weaviate (Docker, port 8080)
#   - finqa_retrieval_adapter (FastAPI, port 3000)
#   - finqa_generator_adapter (FastAPI, port 8000 — calls Bedrock Kimi K2.5)
#   - VERL PPO training on GPUs 1-7
#
# Prerequisites: aws_provision.sh has been run.

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

BOOTSTRAP=$(mktemp)
cat > "$BOOTSTRAP" <<EOF
#!/bin/bash
exec > /var/log/s3-paper-bootstrap.log 2>&1
set -e

echo "[bootstrap] start: \$(date)"
cd /home/ubuntu

# 1. Pull code tarballs from S3 (pitfall #12: private repo workaround)
sudo -u ubuntu bash -c "mkdir -p /home/ubuntu/fin-qa-research /home/ubuntu/s3"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/fin-qa-code.tar.gz /home/ubuntu/fin-qa-code.tar.gz"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/s3-paper-code.tar.gz /home/ubuntu/s3-paper-code.tar.gz"
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && tar -xzf /home/ubuntu/fin-qa-code.tar.gz"
sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && tar -xzf /home/ubuntu/s3-paper-code.tar.gz"
sudo -u ubuntu bash -c "mkdir -p /home/ubuntu/fin-qa-research/dataset"

# 2. uv-based finqa venv (pitfall #7-8)
sudo -u ubuntu bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && /home/ubuntu/.local/bin/uv sync --no-dev"
sudo -u ubuntu bash -c "source /home/ubuntu/fin-qa-research/.venv/bin/activate && /home/ubuntu/.local/bin/uv pip install fastapi uvicorn pyarrow boto3 'transformers>=4.51'"

# 3. Conda s3 env (paper's stack)
sudo -u ubuntu bash -c "conda env create -f /home/ubuntu/s3/environment_s3.yml || conda env update -f /home/ubuntu/s3/environment_s3.yml"
sudo -u ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate s3 && pip install -e /home/ubuntu/s3"

# 4. Weaviate (pitfall #6: docker compose v2)
sudo -u ubuntu bash -c "cd /home/ubuntu/fin-qa-research && docker compose up -d weaviate-finqa"
sleep 15

# 5. Pull FinQA dataset + naive_correct (if exists)
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/dataset/train.json /home/ubuntu/fin-qa-research/dataset/train.json"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/dataset/dev.json /home/ubuntu/fin-qa-research/dataset/dev.json"
sudo -u ubuntu bash -c "aws s3 cp s3://$S3_BUCKET/dataset/test.json /home/ubuntu/fin-qa-research/dataset/test.json"
sudo -u ubuntu bash -c "mkdir -p /home/ubuntu/s3/data/finqa_s3 && (aws s3 sync s3://$S3_BUCKET/finqa_s3/ /home/ubuntu/s3/data/finqa_s3/ || true)"

# 6. Ingest FinQA into Weaviate
cd /home/ubuntu/fin-qa-research/src/nr-pb-step-4-s3-integrated-policy
sudo -u ubuntu bash -c "WEAVIATE_URL=http://127.0.0.1:8080 SKIP_DOCKER_MANAGEMENT=1 /home/ubuntu/fin-qa-research/.venv/bin/python code/ingest.py --train"

# 7. Spot interrupt handler (pitfall #10: HTTP 200 only)
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

# 8. Start FinQA retrieval adapter (port 3000)
sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && WEAVIATE_URL=http://127.0.0.1:8080 FINQA_REPO_ROOT=/home/ubuntu/fin-qa-research nohup /home/ubuntu/fin-qa-research/.venv/bin/uvicorn integrations.finqa_retrieval_adapter:app --host 0.0.0.0 --port 3000 --workers 2 > /home/ubuntu/retrieval-adapter.log 2>&1 &"

# 9. Start FinQA generator adapter (port 8000)
sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && AWS_REGION=us-east-1 FINQA_REPO_ROOT=/home/ubuntu/fin-qa-research nohup /home/ubuntu/fin-qa-research/.venv/bin/uvicorn integrations.finqa_generator_adapter:app --host 0.0.0.0 --port 8000 --workers 2 > /home/ubuntu/generator-adapter.log 2>&1 &"

sleep 20

# 10. Health check
curl -fsS http://127.0.0.1:3000/health || (echo "retrieval adapter down"; exit 1)
curl -fsS http://127.0.0.1:8000/health || (echo "generator adapter down"; exit 1)

# 11. Prepare FinQA parquet (if not already in S3)
if [ ! -f /home/ubuntu/s3/data/finqa_s3/train_finqa_s3.parquet ]; then
    sudo -u ubuntu bash -c "cd /home/ubuntu/s3 && /home/ubuntu/fin-qa-research/.venv/bin/python integrations/prepare_finqa_dataset.py --finqa-dir /home/ubuntu/fin-qa-research/dataset --out-dir data/finqa_s3 --workers 8"
    sudo -u ubuntu bash -c "aws s3 sync /home/ubuntu/s3/data/finqa_s3/ s3://$S3_BUCKET/finqa_s3/"
fi

# 12. Launch VERL PPO training on GPUs 1-7 (paper §A.2 used 5 GPUs; we have 7 available)
sudo -u ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate s3 && cd /home/ubuntu/s3 && VERL_GPUS=1,2,3,4,5,6,7 TOTAL_STEPS=$TOTAL_STEPS GENERATOR_LLM_URL=http://127.0.0.1:8000/v1/chat/completions RETRIEVER_URL=http://127.0.0.1:3000/retrieve bash scripts/train/train_s3_finqa.sh 2>&1 | tee /home/ubuntu/s3-train.log"

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

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    ${MARKET_OPTS[@]+"${MARKET_OPTS[@]}"} \
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
echo
echo "INSTANCE_ID=$INSTANCE_ID" >> "$RUNTIME_FILE"
echo "INSTANCE_IP=$PUBLIC_IP" >> "$RUNTIME_FILE"
echo "MODE=$MODE" >> "$RUNTIME_FILE"
echo
echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE"
