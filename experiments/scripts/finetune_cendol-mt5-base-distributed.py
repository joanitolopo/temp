import subprocess
import logging
import time
import multiprocessing
import os

import multiprocessing
omp_num_threads = max(1, multiprocessing.cpu_count() // 4)  # Adjust based on system
world_size = 4  # Number of processes or GPUs to use
os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
os.environ["WORLD_SIZE"]= str(world_size)

def main():
    # Define arguments
    model_name_or_path = "indonlp/cendol-mt5-small-inst"
    output_dir = "results/baseline/output/cendol-mt5-small-inst"
    learning_rate = "3e-4"
    batch_size = 16
    grad_accum = 4
    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, min(cpu_count // 2, 4))  # Use half of available CPUs, max 4

    # Logging setup
    logging.basicConfig(level=logging.INFO)

    # Define torchrun options
    nproc_per_node = 2  # Number of GPUs to use
    master_port = 1234  # Unique port number to avoid conflicts

    # Build the command
    command = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "./experiments/finetune_seq2seq.py",
        f"--model_name_or_path={model_name_or_path}",
        f"--output_dir={output_dir}",
        "--overwrite_output_dir",
        f"--learning_rate={learning_rate}",
        "--data_path=joanitolopo/KupangMalay-InstructCorpus-v1",
        "--bf16",
        "--fsdp=full_shard auto_wrap",
        "--fsdp_config=configs/fsdp_config.json",
        f"--per_device_train_batch_size={batch_size}",
        f"--per_device_eval_batch_size={batch_size}",
        f"--gradient_accumulation_steps={grad_accum}",
        "--num_train_epochs=1",
        "--source_max_length=512",
        "--model_max_length=512",
        "--continual_size=1000",
        "--val_set_size=0.2",
        "--save_steps=5000",
        "--eval_steps=5000",
        "--logging_steps=100",
        f"--preprocessing_num_workers={num_workers}",
        f"--dataloader_num_workers={num_workers}",
        "--use_lora=False",
        "--lora_r=64",
        "--lora_alpha=16",
        "--lora_dropout=0.05",
        "--lora_target_modules=q, v",
        "--ddp_find_unused_parameters=False",
        "--save_total_limit=3",
        "--group_by_length",
        "--report_to=none"
    ]

    # Log the command
    logging.info(f"Running command: {' '.join(command)}")

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
