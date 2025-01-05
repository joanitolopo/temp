import os
import logging
import random
from datasets import load_dataset, Dataset, DatasetDict
from prompts import prompt_instruction

hf_token = os.getenv("HF_TOKEN")

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def main():
    logger = setup_logging()
    logger.info("Memuat dataset KupangMalay-ParallelCorpus...")
    
    # Memuat dataset dengan token akses
    parallel_data = load_dataset(
        "joanitolopo/KupangMalay-ParallelCorpus-v1", 
        token=hf_token
    )
    logger.info("Dataset berhasil dimuat.")

    instruction_data_train = []
    instruction_data_test = []

    # Proses data latih
    logger.info("Memulai proses data latih...")
    for data in parallel_data['train']:
        prompt = random.choice(prompt_instruction)
        instruction = prompt.format(SOURCE="Indonesia", TARGET="Melayu Kupang", INPUT=data["ind"])
        instruction_data_train.append({
            "prompt": prompt,
            "input": instruction,
            "output": data["mkn"]
        })
    logger.info(f"Data latih selesai diproses. Total: {len(instruction_data_train)} instruksi.")

    # Proses data uji
    logger.info("Memulai proses data uji...")
    for data in parallel_data['test']:
        prompt = random.choice(prompt_instruction)
        instruction = prompt.format(SOURCE="Indonesia", TARGET="Melayu Kupang", INPUT=data["ind"])
        instruction_data_test.append({
            "prompt": prompt,
            "input": instruction,
            "output": data["mkn"]
        })
    logger.info(f"Data uji selesai diproses. Total: {len(instruction_data_test)} instruksi.")

    # Membuat dataset
    logger.info("Membuat objek Dataset...")
    train_dataset = Dataset.from_list(instruction_data_train)
    test_dataset = Dataset.from_list(instruction_data_test)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    logger.info("Dataset berhasil dibuat.")

    # Push ke hub
    logger.info("Mengunggah dataset ke Hugging Face Hub...")
    dataset.push_to_hub(
        "joanitolopo/KupangMalay-InstructCorpus-v1", 
        private=True, 
        token=hf_token
    )
    logger.info("Dataset berhasil diunggah ke Hugging Face Hub.")

if __name__ == "__main__":
    main()
