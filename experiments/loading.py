from utils import save_keywords_to_cache, load_keywords_from_cache, process_instruction
from gensim.models import KeyedVectors
from keybert import KeyBERT
from transformers import pipeline
from datasets import load_dataset, Dataset, DatasetDict
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_token = os.getenv("HF_TOKEN")


def load_word_embeddings(logger, path):
    """Loads FastText word embeddings."""
    logger.info("Loading FastText word embeddings...")
    return KeyedVectors.load_word2vec_format(path)

def load_parallel_dataset(logger):
    """Downloads and loads the parallel dataset."""
    logger.info("Downloading Parallel Dataset...")
    return load_dataset("joanitolopo/KupangMalay-ParallelCorpus-v1", token=hf_token)

def extract_or_load_keywords(logger, data, cache_file, ngram_range=(1, 1), top_n=5, diversity=0.7):
    """Extracts keywords using KeyBERT."""
    if os.path.exists(cache_file):
        logger.info("Loading keywords from cache...")
        return load_keywords_from_cache(cache_file)
    
    logger.info("Initializing KeyBERT with IndoBERT model...")
    indobertmodel = pipeline("feature-extraction", model="indobenchmark/indobert-large-p2", device=device)
    model = KeyBERT(model=indobertmodel)
    
    logger.info("Extracting keywords and saving to cache...")
    keywords = model.extract_keywords(data, keyphrase_ngram_range=ngram_range, top_n=top_n, diversity=diversity)
    save_keywords_to_cache(keywords, cache_file)
    return keywords

def generate_instructions(logger, dataset, generator, keywords):
    """Generates instructions based on the dataset and extracted keywords."""
    logger.info("Generating instructions...")
    return process_instruction(dataset, generator, keywords)

def push_to_hub(logger, train_data, test_data, repo_name):
    """Creates a DatasetDict and pushes it to the Hugging Face Hub."""
    logger.info("Saving the dataset to the Hugging Face Hub...")
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset.push_to_hub(repo_name, private=True, token=hf_token)