import os
import gdown
import random
import json
import transformers
from typing import Dict
from tqdm import tqdm
from datasets import load_dataset


current_dir = os.getcwd()

def load_rehearsal_dataset(n_samples=1000, random_seed=42):
  cendol_dset = load_dataset("indonlp/cendol_collection_v2", split="train", trust_remote_code=True)
  sample_cendol_dset = cendol_dset.shuffle(seed=random_seed).take(n_samples)
  return sample_cendol_dset

def extract_phonetic_rules(dictionary):
  rules = {}
  for ind, kup in dictionary:
    # print(ind, kup)
    for i, (ind_char, kup_char) in enumerate(zip(ind, kup)):
      if ind_char != kup_char:
        if ind_char in rules:
          if kup_char not in rules[ind_char]:
              rules[ind_char].append(kup_char)
        else:
          rules[ind_char] = [kup_char]
  return rules


def extract_dictionary():
  with open(f"{current_dir}/source/merged_data.json", "r", encoding='utf-8') as f:
     bilingual_dict = json.load(f)

  dictionary = []
  for entry in bilingual_dict:
    src = random.choice(entry['word']['target'])
    trg = entry['word']['source']
    dictionary.append((src, trg))
  return bilingual_dict, dictionary

def download_and_load_fasttext(logger):
  # URLs for the resources
  url_vec_zip = 'https://drive.google.com/uc?id=1-3tOLyIY5fTUJs2L5CcyYhg82tL2W4TO'
  url_npy = 'https://drive.google.com/uc?id=1WRV0nH5QjvJRyyaRKYMh19KS31t4NEqW'

  # Output file paths
  output_vec_zip = f'{current_dir}/embedding/fasttext.18k.mkn.300.epoch5.uncased.vec.zip'
  output_npy = f'{current_dir}/embedding/fasttext.18k.mkn.300.epoch5.uncased.bin.wv.vectors_ngrams.npy'

  try:
    # Create the embedding directory if it doesn't exist
    logger.info(f"Checking if the '{current_dir}/embedding' directory exists...")
    os.makedirs(f'{current_dir}/embedding', exist_ok=True)
    logger.info(f"Directory {current_dir}/embedding' is ready.")

    # Download files only if they do not already exist
    if not os.path.exists(output_vec_zip):
        logger.info(f"File not found: {output_vec_zip}. Starting download...")
        gdown.download(url_vec_zip, output_vec_zip, quiet=False)
        logger.info(f"Download completed: {output_vec_zip}")
    else:
        logger.info(f"File already exists: {output_vec_zip}. Skipping download.")

    if not os.path.exists(output_npy):
        logger.info(f"File not found: {output_npy}. Starting download...")
        gdown.download(url_npy, output_npy, quiet=False)
        logger.info(f"Download completed: {output_npy}")
    else:
        logger.info(f"File already exists: {output_npy}. Skipping download.")

  except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
  
  return output_vec_zip, output_npy

# Fungsi untuk memproses data dengan berbagai jenis instruksi
def process_instruction(parallel_data, generator, keywords):
  instruction_data = []
  for idx, data in enumerate(tqdm(parallel_data, desc="Processing data")):
    src = data["ind"]
    trg = data['mkn']

    # Precompute instructions
    w_star = generator.get_special_word(src, keywords=keywords[idx], n_highest=5)
    contextual_inst = generator.generate_contextual_inst(src, w_star)
    semantic_inst = generator.generate_semantic_inst(src, w_star)
    keyword_inst = generator.generate_keyword_inst(src, w_star, n_similar=3)
    list_group_label_inst = generator.generate_list_group_label_inst(src, w_star, n_nearest=2)

    # Tambahkan ke daftar hasil
    instruction_data.extend([
        {"input": src, "prompt": contextual_inst, "output": trg},
        {"input": src, "prompt": semantic_inst, "output": trg},
        {"input": src, "prompt": keyword_inst, "output": trg},
        {"input": src, "prompt": list_group_label_inst, "output": trg}
      ])
  return instruction_data

def ensure_cache_directory(cache_dir = "cache"):
    """Ensures that the cache directory exists."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

def save_keywords_to_cache(keywords, cache_dir):
    """Saves extracted keywords to the cache directory."""
    with open(cache_dir, "w") as f:
        json.dump(keywords, f)

def load_keywords_from_cache(cache_dir):
    """Loads extracted keywords from the cache directory."""
    with open(cache_dir, "r") as f:
        return json.load(f)
    
# Copied from https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
