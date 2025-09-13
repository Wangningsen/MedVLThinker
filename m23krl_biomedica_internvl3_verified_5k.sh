#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
MODEL=/data3/nwang60/model/Qwen2.5-VL-MedVLThinker-3B-RL_m23k
ENGINE=vllm

# —— 载入 .env（WANDB/HF_TOKEN 等） ——
set -a && source .env && set +a

mkdir -p /data3/nwang60/tmp/{hf_cache,hf_datasets,ray_tmp,ray_spill,trainset}

export HF_HOME=/data3/nwang60/tmp/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=/data3/nwang60/tmp/hf_datasets
export HF_DATASETS_DISABLE_MEMORY_MAPPING=1
export ARROW_USE_MMAP=0

export RAY_TMPDIR=/data3/nwang60/tmp/ray_tmp
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/data3/nwang60/tmp/ray_spill"}}'

# —— NCCL 调试（保留这两项足够） ——
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1



# —— 训练 ——
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/data3/nwang60/dataset/MVT-synthesis/biomedica_internvl3_verified_5k_verl/train.parquet \
  data.val_files=/data3/nwang60/dataset/MVT-synthesis/MedVLThinker-Eval_verl/test.parquet \
  data.shuffle=True \
  data.train_batch_size=48 \
  data.max_prompt_length=80000 \
  data.filter_overlong_prompts=False \
  data.max_response_length=4096 \
  data.truncation='error' \
  data.image_key=images \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=24 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='Medvlthinker_synthesis' \
  trainer.experiment_name='m23krl_3b_biomedica_internvl3_verified_5k_grpo' \
  trainer.n_gpus_per_node=6 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.total_epochs=1 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.max_critic_ckpt_to_keep=3 \
  custom_reward_function.path=./train/my_reward.py "$@"




