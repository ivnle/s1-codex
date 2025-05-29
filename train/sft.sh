# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
uid="$(date +%Y%m%d_%H%M%S)"
# base_model="Qwen/Qwen2.5-0.5B-Instruct"
# base_model="Qwen/Qwen2.5-1.5B-Instruct"
# base_model="Qwen/Qwen2.5-3B-Instruct"
base_model="Qwen/Qwen2.5-7B-Instruct"
# base_model="Qwen/Qwen2.5-14B-Instruct"
# base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
# gpu_count=$(nvidia-smi -L | wc -l)
gpu_count=1
push_to_hub=true
cache_dir="/trunk/model-hub" # Define the cache directory
wandb_project="s1-codex" # Define your W&B project
wandb_entity="ivnle"  # Define your W&B entity (username or team)
logging_steps=10
# block_size=32768
# block_size=512
# block_size=2048
# block_size=4096
# block_size=6144
block_size=8192

# Dataset statistics logging
log_dataset_stats=true      # set to false to skip stats computation

# LORA parameters
use_lora=true # Set to true to enable LORA
lora_r=8
lora_alpha=16
lora_dropout=0.05
# Example: lora_target_modules_str="q_proj,v_proj" # Uncomment and set to override defaults in python script
# If lora_target_modules_str is not set or empty, the script will use defaults or try to infer.

# ------------------------ QLoRA parameters ------------------------
use_qlora=true                  # Set to true to activate 4-bit QLoRA
qlora_compute_dtype="bf16"       # "bf16" or "fp16"
# If QLoRA is enabled we must also enable LoRA adapters                                   
if [ "${use_qlora}" = true ]; then
    use_lora=true
fi

# ---------------- Distributed backend -----------------
dist_backend="ddp"                      # "ddp" (default), "fsdp", or "single"
fsdp_policy="full_shard auto_wrap"      # Used only when dist_backend="fsdp"
fsdp_config_file="train/fsdp_config_qwen.json"

# QLoRA is not compatible with FSDP → force DDP
if [ "${use_qlora}" = true ] && [ "${dist_backend}" = "fsdp" ]; then
  echo "[WARN] QLoRA cannot be combined with FSDP; falling back to DDP." >&2
  dist_backend="ddp"
fi

# ---------- Build informative checkpoint / HuggingFace repo name ----------
# Short model id (strip path)
base_model_short=$(basename "${base_model}")

# Parameter-count tag: grabs the first “…B” token such as 0.5B / 7B / 32B
param_tag=$(echo "${base_model_short}" | grep -o -E '[0-9]+(\.[0-9]+)?B' | head -n1)
[ -z "${param_tag}" ] && param_tag="unk"

# PEFT tag
if [ "${use_qlora}" = true ]; then
  peft_tag="qlora"
elif [ "${use_lora}" = true ]; then
  peft_tag="lora"
else
  peft_tag="full"
fi

# Combine: <params>_<peft>_<backend>_<timestamp>
repo_name="${param_tag}_${peft_tag}_${dist_backend}_${uid}"

# Final output directory (also becomes HF repo name when push_to_hub=true)
output_dir="ckpts/${repo_name}"

# -------- Optional explicit checkpoint directory --------
checkpoint_dir="/graft3/checkpoints/ivanlee/s1-codex"          # e.g. "my_ckpts/run42" ; leave empty to use default

# If the user requests single-GPU, override world-size
if [ "${dist_backend}" = "single" ]; then
  gpu_count=1
fi

cmd=(torchrun --nproc-per-node ${gpu_count} --master_port 12344 train/sft.py \
     --block_size=${block_size} \
     --per_device_train_batch_size=${micro_batch_size} \
     --per_device_eval_batch_size=${micro_batch_size} \
     --gradient_accumulation_steps=${gradient_accumulation_steps} \
     --num_train_epochs=${epochs} \
     --train_file_path="simplescaling/s1K_tokenized" \
     --model_name=${base_model} \
     --cache_dir=${cache_dir} \
     --wandb_project=${wandb_project} \
     --wandb_entity=${wandb_entity} \
     --warmup_ratio=0.05 \
     --bf16=True \
     --eval_strategy="no" \
     --logging_steps=${logging_steps} \
     --save_strategy="no" \
     --lr_scheduler_type="cosine" \
     --learning_rate=${lr} \
     --weight_decay=${weight_decay} \
     --adam_beta1=0.9 \
     --adam_beta2=0.95 \
     --output_dir="${output_dir}" \
     --push_to_hub=${push_to_hub} \
     --save_only_model=True \
     --use_lora=${use_lora} \
     --lora_r=${lora_r} \
     --lora_alpha=${lora_alpha} \
     --lora_dropout=${lora_dropout} \
     --use_qlora=${use_qlora} \
     --qlora_compute_dtype=${qlora_compute_dtype} \
     --log_dataset_stats=${log_dataset_stats})

# Add FSDP flags only when requested
if [ "${dist_backend}" = "fsdp" ]; then
  cmd+=(--fsdp="${fsdp_policy}" --fsdp_config="${fsdp_config_file}")
fi

# Add checkpoint_dir flag if set
if [ -n "${checkpoint_dir}" ]; then
  cmd+=(--checkpoint_dir="${checkpoint_dir}")
fi

# Launch
"${cmd[@]}"
    # To specify target modules from shell, uncomment and add:
    # --lora_target_modules="${lora_target_modules_str}" \
    # --gradient_checkpointing=True \ Enable gradient checkpointing for efficient memory usage with 8 H100 GPUs.
    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'

