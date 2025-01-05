import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNING'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set environment variables untuk mengarahkan cache ke lokasi baru
# save memory
# os.environ['HF_HOME'] = 'D:/.cache/huggingface'
# os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/.cache/huggingface/hub'
# # os.environ['TRANSFORMERS_CACHE'] = 'D:/.cache/huggingface/hub'
# os.environ['HF_DATASETS_CACHE'] = 'D:/.cache/huggingface/datasets'

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)

import sys
import logging
import datasets

import torch


import transformers
from typing import Optional
from accelerate import Accelerator
from dataclasses import dataclass, field

from utils import load_rehearsal_dataset, smart_tokenizer_and_embedding_resize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser, BitsAndBytesConfig
from datasets import Dataset, load_dataset, interleave_datasets


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    # Base model paramaters
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8-bit mode"})

    # LoRA parameters
    use_lora: bool = field(default=False, metadata={"help": "Use LoRA quantization"})
    lora_r: int = field(default=8, metadata={"help": "LoRA r parameter"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate"})
    lora_target_modules: str = field(default="q, v", metadata={"help": "LoRA target modules"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the data directory"})
    source_max_length: Optional[int] = field(default=256, metadata={"help": "Max length of the source sequence. Sequences will be right padded (and possibly truncated)."})
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Max length of the target sequence. Sequences will be right padded (and possibly truncated)."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    continual_size: Optional[int] = field(default=1000, metadata={"help": "The number of examples to use in experience replay."})
    val_set_size: Optional[float] = field(default=0.2, metadata={"help": "The number of examples to sample from the dataset for validation."})

@dataclass
class LiusTrainingArguments(Seq2SeqTrainingArguments):
    optim: str = field(default="adamw_torch", metadata={"help": "The optimizer to use"})
    fp16: bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision training"})
    evaluation_strategy: str = field(default="steps", metadata={"help": "The evaluation strategy to use"})
    save_strategy: str = field(default="steps", metadata={"help": "The save strategy to use"})
    wandb_project: Optional[str] = field(default="lius_project", metadata={"help": "The name of the wandb project"})
    push_to_hub: bool = field(default=False, metadata={"help": "Whether to push the model to the hub"})


def train():
    accelerator = Accelerator(split_batches=False)

    parser = HfArgumentParser((ModelArguments, DataArguments, LiusTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log in each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" 
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "wandb" in training_args.report_to:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=model_args.load_in_8bit)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=quantization_config)
    model.config.use_cache = False

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="right", use_fast=True)

    if model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()
    
    if model_args.use_lora:
        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    
    raw_datasets = load_dataset(data_args.data_path, token=os.getenv("HF_TOKEN"))
    sample_dset = load_rehearsal_dataset(n_samples=data_args.continual_size)
    sample_dset = Dataset.from_list(list(sample_dset))
    raw_datasets["train"] = interleave_datasets([sample_dset, raw_datasets["train"]], stopping_strategy="all_exhausted")

    # determine model_max_length and source_max_length for truncation
    source_max_length = data_args.source_max_length
    model_max_length = data_args.model_max_length

    def generate_and_tokenize_prompt(data_point):
        user_prompt = f'{data_point["input"]} '
        target_text = f'{data_point["output"]}'
        source_ids = tokenizer(text=user_prompt, truncation=True, max_length=source_max_length)["input_ids"]
        target_ids = tokenizer(text_target=target_text, truncation=True, max_length=model_max_length-source_max_length)["input_ids"]
        return {"input_ids":source_ids, "labels":target_ids}

    # sampling
    train_val_data = raw_datasets["train"].train_test_split(test_size=data_args.val_set_size, shuffle=True, seed=42) 
    
    # splitting
    train_data = train_val_data["train"].map(
        generate_and_tokenize_prompt,
        remove_columns=train_val_data['test'].column_names,
        desc="Processing training data",
    )
    val_data = train_val_data["test"].map(
        generate_and_tokenize_prompt,
        remove_columns=train_val_data['test'].column_names,
        desc="Processing validation data",
    )

    # happy traing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, 
                                                          pad_to_multiple_of=8,
                                                          return_tensors="pt",
                                                          padding=True)
    )

    if model_args.use_lora:
        old_state_dict = model.state_dict()
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict)
        ).__get__(model, type(model))
    
    trainer.train()

    model = accelerator.unwrap_model(model)
    model.save_pretrained(training_args.output_dir,
                          is_main_process=accelerator.is_main_process,
                          save_function=accelerator.save)
    

if __name__ == "__main__":
    train()



