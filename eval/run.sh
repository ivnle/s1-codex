export HF_HUB_CACHE="/trunk/model-hub"
export PROCESSOR="gpt-4o-mini"
# TOKENIZER_AND_PRETRAINED_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
# TOKENIZER_AND_PRETRAINED_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# TOKENIZER_AND_PRETRAINED_MODEL="Qwen/Qwen2.5-3B-Instruct"
# TOKENIZER_AND_PRETRAINED_MODEL="Qwen/Qwen2.5-7B-Instruct"
# TOKENIZER_AND_PRETRAINED_MODEL="Qwen/Qwen2.5-14B-Instruct"
# TOKENIZER_AND_PRETRAINED_MODEL="ivnle/0.5B_full_fsdp_bs8192_20250528_222808"
# TOKENIZER_AND_PRETRAINED_MODEL="ivnle/0.5b-full-sft"
# TOKENIZER_AND_PRETRAINED_MODEL="ivnle/3B_full_fsdp_bs8192_20250528_224745"
TOKENIZER_AND_PRETRAINED_MODEL="ivnle/7B_full_fsdp_bs8192_20250529_061725"
DTYPE="float32"
TENSOR_PARALLEL_SIZE=1

lm_eval \
  --model vllm \
  --model_args pretrained=${TOKENIZER_AND_PRETRAINED_MODEL},tokenizer=${TOKENIZER_AND_PRETRAINED_MODEL},dtype=${DTYPE},tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
  --tasks aime24_figures,aime24_nofigures,openai_math,gpqa_diamond_openai \
  --batch_size auto \
  --apply_chat_template \
  --output_path qwen \
  --log_samples \
  --gen_kwargs "max_gen_toks=8192"
