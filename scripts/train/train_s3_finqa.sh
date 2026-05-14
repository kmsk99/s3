#!/bin/bash
# s3 paper PPO training on FinQA.
#
# Hyperparameters: scripts/train/train_s3.sh (s3 paper §A.2) with two
# substitutions only:
#   - dataset path → FinQA parquet
#   - reward_score data_source → finqa_exec_acc (numeric match)
#
# Requires:
#   - finqa_retrieval_adapter running at retriever.url (port 3000)
#   - finqa_generator_adapter running at generator_llm OpenAI base URL
#   - Weaviate populated with FinQA chunks
#   - prepare_finqa_dataset.py has produced data/finqa_s3/{train,valid,test}_finqa_s3.parquet
#
# Adapt to p4d.24xlarge: 8 GPUs available, but we leave GPU 0 for adapters and
# allocate 7 GPUs to VERL (matching paper's 5-GPU setup). Adjust as needed.

data_name=finqa
RANDOM_SEED=${1:-42}

# GPU 0 reserved for retrieval/generator adapters; GPUs 1-7 for VERL.
export CUDA_VISIBLE_DEVICES=${VERL_GPUS:-1,2,3,4,5,6,7}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

export DATA_DIR=data/finqa_s3
WAND_PROJECT="FinQA-s3"

# Paper §A.2 uses Qwen2.5-7B-Instruct. We match.
export BASE_MODEL=${BASE_MODEL:-'Qwen/Qwen2.5-7B-Instruct'}
export EXPERIMENT_NAME="s3_finqa_8_3_3_${RANDOM_SEED}"
export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_finqa_s3.parquet \
    data.val_files=$DATA_DIR/valid_finqa_s3.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=120 \
    data.val_batch_size=15 \
    data.max_prompt_length=8000 \
    data.max_response_length=500 \
    data.max_start_length=2000 \
    data.max_obs_length=1400 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=30 \
    actor_rollout_ref.actor.ppo_micro_batch_size=15 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=30 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=30 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.01 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    critic.ppo_micro_batch_size=10 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1500 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=4 \
    trainer.total_training_steps=${TOTAL_STEPS:-20} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    +data.random_seed=$RANDOM_SEED \
    max_turns=3 \
    +generator_llm="${GENERATOR_LLM_URL:-http://127.0.0.1:8000/v1/chat/completions}" \
    +output_context_dir="data/output_sequences_finqa_$EXPERIMENT_NAME" \
    retriever.url="${RETRIEVER_URL:-http://127.0.0.1:3000/retrieve}" \
    retriever.topk=8 \
    2>&1 | tee train_logs/$EXPERIMENT_NAME.log
