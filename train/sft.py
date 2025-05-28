import os
import torch
from transformers import BitsAndBytesConfig          # NEW – only used when use_qlora
from dataclasses import dataclass, field, asdict
from typing import Optional
import itertools
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging

# ----------------------------------------------------------------------
# Helpers to know if we are the global-rank-0 / main process.
# Works with torchrun (env RANK) and also when not in distributed mode.
# ----------------------------------------------------------------------
def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    return rank is None or int(rank) == 0

log_level = logging.INFO if _is_main_process() else logging.ERROR
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,           # overwrite any pre-existing handlers
)
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
from peft import LoraConfig
import numpy as np              # NEW

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default=None)
    wandb_entity: Optional[str] = field(default=None)
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    cache_dir: Optional[str] = field(default=None)
    dagger: bool = field(default=False)
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LORA."})
    use_qlora: bool = field(default=False,
        metadata={"help": "Use QLoRA (4-bit base model + LoRA adapters) instead of full-precision base"})
    qlora_compute_dtype: str = field(
        default="bf16", metadata={"help": "'bf16' or 'fp16' – dtype used for 4-bit dequant compute"})
    lora_r: int = field(default=8, metadata={"help": "LORA attention dimension."})
    lora_alpha: int = field(default=16, metadata={"help": "LORA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LORA dropout."})
    lora_target_modules: Optional[list[str]] = field(
        default=None, metadata={"help": "List of module names to apply LORA to (e.g., 'q_proj,v_proj'). Defaults will be used if None."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory in which to save checkpoints / final model. "
                          "If provided, overrides the Trainer `output_dir`."},
    )
    log_dataset_stats: bool = field(
        default=False,
        metadata={"help": "If true, log basic token-length statistics of the dataset before training."},
    )

    def __post_init__(self):
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    # ------------------------------------------------------------------
    # Resolve final checkpoint directory (defaults to Trainer's output_dir)
    # ------------------------------------------------------------------
    checkpoint_dir = config.checkpoint_dir or args.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    # make sure the Trainer itself also points there
    args.output_dir = checkpoint_dir
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # -------------------- MODEL LOADING --------------------
    if config.use_qlora:
        # 4-bit NF4 quantisation + double quant (QLoRA)
        compute_dtype = torch.bfloat16 if config.qlora_compute_dtype.lower() == "bf16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        # make sure LoRA is actually enabled for QLoRA
        config.use_lora = True
    elif "70B" in config.model_name:
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
            "cache_dir": config.cache_dir,
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name, cache_dir=config.cache_dir
        )

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True, cache_dir=config.cache_dir)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size

    # ------------------------------------------------------------------
    # (Optional) dataset statistics
    # ------------------------------------------------------------------
    if config.log_dataset_stats:
        logging.info("Computing dataset token statistics ...")
        text_field = args.dataset_text_field  # typically "text"

        def _example_len(ex):
            # If already tokenized, use it; otherwise tokenize now.
            if "input_ids" in ex and isinstance(ex["input_ids"], (list, tuple)):
                return len(ex["input_ids"])
            return len(
                tokenizer(
                    ex[text_field],
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            )

        train_lengths = [ _example_len(ex) for ex in dataset["train"] ]

        total_examples = len(train_lengths)
        total_tokens   = sum(train_lengths)
        min_len        = min(train_lengths)
        max_len        = max(train_lengths)
        mean_len       = total_tokens / total_examples if total_examples else 0

        logging.info(
            "Train split ‖ examples: %d ‖ total tokens: %d ‖ "
            "min: %d ‖ max: %d ‖ mean: %.2f",
            total_examples,
            total_tokens,
            min_len,
            max_len,
            mean_len,
        )

        # ---------- NEW: CDF for train split ----------
        # 10 automatically-chosen bins; change "auto" or bin count as desired.
        train_hist_counts, train_hist_edges = np.histogram(train_lengths, bins="auto")
        cum_counts = np.cumsum(train_hist_counts)
        cum_frac   = cum_counts / cum_counts[-1]
        logging.info("Train token-length CDF (≤ bin_end : cum_count  cum_pct)")
        for right_edge, cum_ct, cum_p in zip(train_hist_edges[1:], cum_counts, cum_frac):
            logging.info("  ≤ %6d : %7d  (%.2f%%)", int(right_edge), int(cum_ct), 100 * cum_p)

        if "test" in dataset:
            test_lengths = [ _example_len(ex) for ex in dataset["test"] ]
            test_total_examples = len(test_lengths)
            test_total_tokens   = sum(test_lengths)
            test_min_len        = min(test_lengths)
            test_max_len        = max(test_lengths)
            test_mean_len       = test_total_tokens / test_total_examples if test_total_examples else 0

            logging.info(
                "Test split ‖ examples: %d ‖ total tokens: %d ‖ "
                "min: %d ‖ max: %d ‖ mean: %.2f",
                test_total_examples,
                test_total_tokens,
                test_min_len,
                test_max_len,
                test_mean_len,
            )

            # ---------- NEW: CDF for test split ----------
            test_hist_counts, test_hist_edges = np.histogram(test_lengths, bins="auto")
            test_cum_counts = np.cumsum(test_hist_counts)
            test_cum_frac   = test_cum_counts / test_cum_counts[-1]
            logging.info("Test token-length CDF (≤ bin_end : cum_count  cum_pct)")
            for right_edge, cum_ct, cum_p in zip(test_hist_edges[1:], test_cum_counts, test_cum_frac):
                logging.info("  ≤ %6d : %7d  (%.2f%%)", int(right_edge), int(cum_ct), 100 * cum_p)

    peft_config = None
    if config.use_lora:
        target_modules = config.lora_target_modules
        if target_modules is None:
            if "Qwen" in config.model_name:
                # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]                
            elif "Llama" in config.model_name:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                logging.warning(
                    "LoRA target modules not specified and no default for model type. "
                    "LoRA might not be effective. You can specify target_modules like --lora_target_modules 'q_proj,v_proj'"
                )
                target_modules = []

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator,
        peft_config=peft_config,
    )

    # Log trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in trainer.model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    logging.info(f"Model parameter dtype: {next(trainer.model.parameters()).dtype}")
    trainer.train()
    trainer.save_model(output_dir=checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
