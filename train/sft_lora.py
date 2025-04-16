import argparse
import os
import math
import sys
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils import tensorboard
import warnings

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    TaskType,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import PartialState, prepare_pippy

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
 #from utils.utils import set_random_seed

os.environ["WANDB_PROJECT"] = "RocketEval-sLLMs"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--context_window",
    type=int,
    default=8012,
    help="Context Window Size for LLaMA3 (defaults to 8012, maximum 8012)",
)
parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA")
parser.add_argument("--epochs", type=int, default=4, help="# of Epochs")
parser.add_argument(
    "--per_device_batch_size", type=int, default=1, help="# of Batches per GPU"
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=4,
    help="# of examples to accumulate gradients before taking a step",
)
parser.add_argument(
    "--checkpointing_ratio",
    type=float,
    default=0.01,
    help="Percentage of Epochs to be Completed Before a Model Saving Happens",
)
parser.add_argument("--fp16", action="store_true", default=True, help="whether or not to use FP16")
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default="RocketEval-sLLMs-v0_3",
    help="Wandb Logging Name for this Training Run",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="use very small dataset(~100) to validate the fine-tuning process",
)
parser.add_argument(
    "--ignore_warning",
    type=bool,
    default=True
)
args = parser.parse_args()
if args.ignore_warning:
    warnings.filterwarnings("ignore", category=UserWarning)


# set_random_seed(42)
PAD_TOKEN = "<|pad|>"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
tokenizer.pad_token = PAD_TOKEN
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
tokenizer.padding_side = "right"
tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<|start_header_id|>", "<|end_header_id|>",
        "<|begin_of_query|>", "<|end_of_query|>", 
        "<|begin_of_reference_response|>", "<|end_of_reference_response|>",
    ]
})

print(tokenizer.special_tokens_map)

response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    response_template, tokenizer=tokenizer
)  # Generation Part 제외한 Instruction, FewShot Example 부분은 -100으로 마스킹하여 파인튜닝 성능개선


base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=args.lora_rank,
    use_rslora=True,
    bias="none",
    init_lora_weights="gaussian",
    task_type=TaskType.CAUSAL_LM,
)

if args.test:
    train_dir = "data/train_data/trainset_merged_train.jsonl"
    validation_dir = "data/train_data/trainset_merged_val.jsonl"
else:
    train_dir = "data/train_data/lmsys_chat_1m_en_expand_gold_final.jsonl"
    validation_dir = "data/train_data/250407_meeting/trainset_merged_val.jsonl"  # "data/train_data/lmsys_chat_1m_en_expand_gold_final_eval.jsonl"

dataset = load_dataset(
    "json", data_files={"train": train_dir, "validation": validation_dir}
)

peft_model = get_peft_model(base_model, lora_config)

# 4: Train the PeftModel (same way as training base model)
sft_config = SFTConfig(
    output_dir="./Models/v0_3",
    dataset_text_field="text",
    max_seq_length=args.context_window,
    num_train_epochs=args.epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_train_batch_size=args.per_device_batch_size,
    per_device_eval_batch_size=args.per_device_batch_size,
    fp16=args.fp16,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    save_strategy="steps",
    save_steps=args.checkpointing_ratio,
    eval_strategy="steps",
    eval_steps=args.checkpointing_ratio,
    report_to="wandb",
    run_name=args.wandb_run_name,
    logging_steps=1,
)
trainer = SFTTrainer(
    peft_model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    # tokenizer=tokenizer,
    data_collator=collator,
)

peft_model.print_trainable_parameters()

# 5: Start Training
trainer.train()

trainer.save_model()