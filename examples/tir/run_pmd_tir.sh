#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENKIMI_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
VERL_DIR="$OPENKIMI_ROOT/verl"
CONFIG_PATH="$VERL_DIR/verl/trainer/config"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$OPENKIMI_ROOT/examples/tir/sandbox/local_sandbox_tool_config.yaml}"
TIR_REWARD_MODULE_PATH="${TIR_REWARD_MODULE_PATH:-$OPENKIMI_ROOT/openkimi/tir/tir_reward_manager.py}"
TIR_AGENT_LOOP_CONFIG_PATH="${TIR_AGENT_LOOP_CONFIG_PATH:-$OPENKIMI_ROOT/openkimi/tir/tir_agent_loop_config.yaml}"

export PYTHONPATH="$OPENKIMI_ROOT:$VERL_DIR:${PYTHONPATH:-}"
export LOCAL_SANDBOX_URL="${LOCAL_SANDBOX_URL:-http://127.0.0.1:12345/faas/sandbox/}"
export VLLM_USE_V1=1
export VERL_HERMES_FALLBACK="${VERL_HERMES_FALLBACK:-1}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)"
fi

TRAIN_FILES="${TRAIN_FILES:-['/mnt/project/yuheng/dataset/deepscaler/train.parquet']}"
VAL_FILES="${VAL_FILES:-['/mnt/project/yuheng/dataset/aime/aime24.parquet','/mnt/project/yuheng/dataset/aime/aime25.parquet']}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8000}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.6}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
NGPUS="${NGPUS:-8}"
NNODES="${NNODES:-1}"
TEST_FREQ="${TEST_FREQ:-20}"
TOTAL_STEPS="${TOTAL_STEPS:-400}"
SAVE_FREQ="${SAVE_FREQ:-20}"
N_VAL="${N_VAL:-32}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"

# ── Rollout mismatch correction (rollout IS) ──────────────────────────────────
ROLLOUT_IS_LEVEL="${ROLLOUT_IS_LEVEL:-sequence}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-5.0}"
ROLLOUT_IS_BATCH_NORM="${ROLLOUT_IS_BATCH_NORM:-False}"

# ── Void-turn masking ─────────────────────────────────────────────────────────
MASK_VOID_TURNS="${MASK_VOID_TURNS:-False}"

# ── PMD algorithm ─────────────────────────────────────────────────────────────
ADV_ESTIMATOR="${ADV_ESTIMATOR:-rloo}"    # rloo = PMD-mean, ploo = PMD-partition
POLICY_LOSS_MODE="${POLICY_LOSS_MODE:-opmd}"
PMD_TAU="${PMD_TAU:-0.01}"
PMD_REWARD_LB="${PMD_REWARD_LB:-null}"
PMD_REWARD_UB="${PMD_REWARD_UB:-null}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
ACTOR_LR="${ACTOR_LR:-1e-6}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
MODEL_NAME_FOR_EXP="${MODEL_PATH##*/}"
TRAIN_DATASET_TAG="$(
  echo "${TRAIN_FILES}" \
  | sed -E "s#/mnt/project/yuheng/dataset/##g; s#\\.parquet##g" \
  | tr -d "[]'\" " \
  | tr '/' '-' \
  | tr ',' '+'
)"
TRAIN_DATASET_TAG="${TRAIN_DATASET_TAG:-unknown_dataset}"
PROJECT_NAME="${PROJECT_NAME:-openkimi_tir}"
EXP_NAME="${EXP_NAME:-pmd_tir_${MODEL_NAME_FOR_EXP}_${TRAIN_DATASET_TAG}_bs${TRAIN_BATCH_SIZE}_ppobs${PPO_MINI_BATCH_SIZE}_n${ROLLOUT_N}_t${MAX_TURNS}_${ADV_ESTIMATOR}_tau${PMD_TAU}_is${ROLLOUT_IS_LEVEL}_th${ROLLOUT_IS_THRESHOLD}_maskvoid${MASK_VOID_TURNS}}"

THINKING_ARGS=()
if [[ "${MODEL_PATH}" == *"Qwen3"* ]]; then
  THINKING_ARGS+=(+data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}")
fi

LOG_DIR="${LOG_DIR:-$OPENKIMI_ROOT/logs}"
CKPT_ROOT_DIR="${CKPT_ROOT_DIR:-/mnt/project/yuheng/exp}"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}_${RUN_TS}.log}"
echo "[log] writing to ${LOG_FILE}"

SANDBOX_PREFLIGHT="${SANDBOX_PREFLIGHT:-1}"
if [[ "${SANDBOX_PREFLIGHT}" == "1" ]]; then
  python3 "$OPENKIMI_ROOT/examples/tir/sandbox/sandbox_smoke_test.py" --url "${LOCAL_SANDBOX_URL}" --timeout 20
fi

python3 -m openkimi.tir.main_tir \
  --config-path="${CONFIG_PATH}" \
  --config-name='ppo_trainer' \
  \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  +algorithm.partition_tau="${PMD_TAU}" \
  +algorithm.partition_reward_lb="${PMD_REWARD_LB}" \
  +algorithm.partition_reward_ub="${PMD_REWARD_UB}" \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is="${ROLLOUT_IS_LEVEL}" \
  algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
  algorithm.rollout_correction.rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORM}" \
  \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  "${THINKING_ARGS[@]}" \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.policy_loss.loss_mode="${POLICY_LOSS_MODE}" \
  +actor_rollout_ref.actor.policy_loss.pmd_tau="${PMD_TAU}" \
  +trainer.tir.mask_void_turns="${MASK_VOID_TURNS}" \
  \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.n="${N_VAL}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tir_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${TIR_AGENT_LOOP_CONFIG_PATH}" \
  \
  critic.enable=False \
  reward_model.enable=False \
  reward_model.reward_manager=tir_dapo \
  reward_model.reward_loop_source=importlib \
  reward_model.reward_loop_module_path="${TIR_REWARD_MODULE_PATH}" \
  reward_model.reward_loop_class_name=TIRDAPORewardManager \
  +reward_model.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
  ++reward_model.reward_kwargs.overlong_buffer_cfg.len=128 \
  ++reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  ++reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
  \
  trainer.critic_warmup=0 \
  trainer.balance_batch=True \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXP_NAME}" \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.val_before_train=False \
  trainer.logger='["console","wandb"]' \
  "$@" 2>&1 | tee "${LOG_FILE}"
