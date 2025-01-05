import os
import subprocess
import logging
import time
import torch

# Cek apakah GPU mendukung bf16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
supports_bf16 = properties.major >= 8 if properties else False
precision = "bf16" if supports_bf16 else "fp16"

# Bersihkan cache GPU
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

def main():
    # Define paths
    model_name_or_path = "indonlp/cendol-mt5-small-inst"
    data_dir = "joanitolopo/KupangMalay-InstructCorpus-v2"
    output_dir = f'results/cendol-mt5-small-inst'
    os.makedirs(output_dir, exist_ok=True)

     # Logging setup
    logging.basicConfig(level=logging.INFO)

    # Build the command
    command = [
        "python", "experiments/finetune_seq2seq.py",
        f"--model_name_or_path={model_name_or_path}",
        f"--output_dir={output_dir}",
        f"--data_path={data_dir}",
        f"--per_device_train_batch_size=2",
        f"--per_device_eval_batch_size=2",
        f"--gradient_accumulation_steps=8",
        f"--num_train_epochs=1",
        f"--torch_empty_cache_steps=500",
        "--learning_rate=2e-4",
        "--overwrite_output_dir=True",
        "--source_max_length=64",
        "--model_max_length=64",
        "--continual_size=1000",
        "--warmup_steps=300",
        "--val_set_size=0.2",
        "--save_steps=2000",
        "--eval_steps=2000",
        "--logging_steps=50",
        "--preprocessing_num_workers=2",
        "--dataloader_num_workers=2",
        "--save_total_limit=3",
        "--gradient_checkpointing",
        f"--{precision}",     
        "--group_by_length",
        "--wandb_project=lius",
        "--report_to=none",
        "--push_to_hub=True",
        f"--hub_token={os.getenv('HF_TOKEN')}"

    ]

    # Measure execution time
    start_time = time.time()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during execution: {e}")
    finally:
        duration = time.time() - start_time
        logging.info(f"Execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
