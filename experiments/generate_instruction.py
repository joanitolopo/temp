import random
import string
import re
import logging
import sys
import os
import torch
import json
from collections import defaultdict
from utils import download_and_load_fasttext, extract_dictionary, extract_phonetic_rules
from difflib import SequenceMatcher
from utils import download_and_load_fasttext, extract_phonetic_rules, ensure_cache_directory

# import promts
from prompts import prompt_instruction, contextual_prompts, semantic_prompts, keyword_prompts, list_group_label_prompts
from loading import load_word_embeddings, load_parallel_dataset, extract_or_load_keywords, push_to_hub, generate_instructions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.getcwd()

class InstructionGenerator:
    def __init__(self, bilingual_dict, w2v_model, phonetic_rules):
        """
        Initializes the InstructionGenerator.

        :param bilingual_dict: A dictionary containing bilingual mappings and example sentences.
        :param w2v_model: A word2vec model to find similar words.
        :param phonetic_rules: A dictionary of phonetic transformation rules.
        """
        self.bilingual_dict = bilingual_dict
        self.w2v_model = w2v_model
        self.phonetic_rules = phonetic_rules

        # get prompt templates
        self.prompt_instruction = prompt_instruction
        self.contextual_prompt = contextual_prompts
        self.semantic_prompt = semantic_prompts
        self.keyword_prompt = keyword_prompts
        self.list_group_label_prompt = list_group_label_prompts

        # track the number of times each prompt type is used
        self.prompt_instruction_counts = defaultdict(int)
        self.contextual_prompt_counts = defaultdict(int)
        self.semantic_prompt_counts = defaultdict(int)
        self.keyword_prompt_counts = defaultdict(int)
        self.list_group_label_prompt_counts = defaultdict(int)
        
    def random_word_prob(self):
        """
        Randomly choose words from bilingual .
        """
        keywords = []
        for _ in range(5):
            bb = random.choice(self.bilingual_dict)['word']['source']
            cc = random.random()
            keywords.append((bb, cc))

        return keywords
    
    def get_special_word(self, sentence, keywords, n_highest=1):
        """
        Extracts the most significant words from a sentence.

        :param sentence: Input sentence to process.
        :param n_highest: Number of highest-ranked keywords to return.
        :keywords: List of keywords in sentences.
        :return: List of top keywords.
        """
        if keywords == []:
            keywords = self.random_word_prob()

        try:
            num_in_sent = int(re.search(r'\d+', sentence).group())
        except:
            num_in_sent = ""

        filtered_keywords = [
            kw for kw in keywords
            if kw[0] != str(num_in_sent) and kw[0] not in string.punctuation
        ]

        if filtered_keywords == []:
            filtered_keywords = self.random_word_prob()

        return sorted(filtered_keywords, key=lambda x: x[1], reverse=True)[:n_highest]
    
    def find_paralel_ind2mkn(self, words):
        """
        Finds the parallel Melayu Kupang word for a indonesia word in the bilingual dictionary.

        :param words: List of words to search for.
        :return: The first matching parallel Melayu Kupang word or the original word.
        """
        for word in words:
            for entry in self.bilingual_dict:
                if word[0] in entry['word']['target']:
                    return entry['word']['source']
        return words[0][0]

    def find_parallel_mkn2ind(self, nearest_words):
        """
        Finds the parallel Indonesia word for list of Melayu Kupang word in the bilingual dictionary..

        :param nearest_words: List of tuples containing words and their similarity scores.
        :return: List of tuples with source and target words.
        """
        result = []
        for word in nearest_words:
            for entry in self.bilingual_dict:
                if word[0] == entry['word']['source']:
                    word_ind = random.choice(entry['word']['target'])
                    result.append((word_ind, word[0]))
                    break
            else:
                result.append((word[0], word[0]))

        return result

    def get_example_sentences(self, word, n):
        """
        Retrieves example sentences containing a given word.

        :param word: Word to search for in the bilingual dictionary.
        :param n: Number of sentences to retrieve.
        :return: List of example sentences.
        """
        sentences = [sentence for entry in self.bilingual_dict for sentence in entry['sentences']['source']]
        related_sentences = [s for s in sentences if word in s.split()]

        if not related_sentences:
            return random.sample(sentences, min(n, len(sentences)))

        if len(related_sentences) < n:
            chosen_sentences = related_sentences + random.sample(sentences, n - len(related_sentences))
            return chosen_sentences

        return random.sample(related_sentences, n)
    
    def phonetic_representation(self, word):
        """
        Converts a word to its phonetic representation using given rules.

        :param word: Input word.
        :return: Phonetic representation of the word.
        """
        phonetic_word = word.lower()
        for src_char, trg_chars in self.phonetic_rules.items():
            for trg_char in trg_chars:
                phonetic_word = phonetic_word.replace(src_char, trg_char)

        return phonetic_word

    def get_similar_sounding_words(self, word, n_similar):
        """
        Retrieves words that sound similar based on phonetic rules.

        :param word: Input word.
        :param n_similar: Number of similar words to retrieve.
        :param phonetic_rules: Dictionary of phonetic transformation rules.
        :return: List of similar-sounding words.
        """
        candidates = [entry['word']['source'] for entry in self.bilingual_dict]
        word_phonetic = self.phonetic_representation(word)
        similar_words = [candidate for candidate in candidates if SequenceMatcher(None, word_phonetic, self.phonetic_representation(candidate)).ratio() > 0.7]

        return similar_words[:n_similar]
    
    def get_prompt_counts(self):
        """
        Retrieves the number of times each prompt type is used.

        :return: Dictionary containing prompt counts.
        """
        return {
            "prompt_instruction": dict(self.prompt_instruction_counts),
            "contextual_prompt": dict(self.contextual_prompt_counts),
            "semantic_prompt": dict(self.semantic_prompt_counts),
            "keyword_prompt": dict(self.keyword_prompt_counts),
            "list_group_label_prompt": dict(self.list_group_label_prompt_counts)}

    def generate_contextual_inst(self, sentence, w_star, n=2):
        """
        Generates a contextual instruction based on a sentence.

        :param sentence: Input sentence.
        :param w_star: A word that represents a sentence.
        :param n: Number of example sentences to include.
        :return: Contextual instruction as a formatted string.
        """
        w_mkn_star = self.find_paralel_ind2mkn(w_star)
        example_sentences = self.get_example_sentences(w_mkn_star, n)
        context = "\n".join([f"- {sent}" for sent in example_sentences])
       
        # track the number of times each prompt type is used
        instruction_template = random.choice(self.prompt_instruction)
        self.prompt_instruction_counts[instruction_template] += 1
        contextual_template = random.choice(self.contextual_prompt)
        self.contextual_prompt_counts[contextual_template] += 1

        instruction = instruction_template.format(SOURCE="Indonesia", INPUT=sentence, TARGET="Melayu Kupang")        
        final_prompt = contextual_template.format(TARGET="Melayu Kupang", CONTEXT=context, INSTRUCTION=instruction)
        return final_prompt
        
    def generate_semantic_inst(self, sentence, w_star, n=2):
        """
        Generates a semantic instruction based on a sentence.

        :param sentence: Input sentence.
        :param w_star: A word that represents a sentence.
        :param n: Number of nearest words to include.
        :return: Semantic instruction as a formatted string.
        """
        w_mkn_star = self.find_paralel_ind2mkn(w_star)
        try:
            nearest_words = self.w2v_model.most_similar(w_mkn_star, topn=n)
        except:
            nearest_words = self.random_word_prob()[:n]
        nearest_words.append((w_mkn_star, 1))
        paralel_n_words = self.find_parallel_mkn2ind(nearest_words)
        context = "\n".join([f"- {src}:{trg}" for src, trg in paralel_n_words])

        # track the number of times each prompt type is used
        instruction_template = random.choice(self.prompt_instruction)
        self.prompt_instruction_counts[instruction_template] += 1
        semantic_template = random.choice(self.semantic_prompt)
        self.semantic_prompt_counts[semantic_template] += 1

        instruction = instruction_template.format(SOURCE="Indonesia", INPUT=sentence, TARGET="Melayu Kupang")
        final_prompt = semantic_template.format(SOURCE="Indonesia", TARGET="Melayu Kupang", CONTEXT=context, INSTRUCTION=instruction)
        return final_prompt

    def generate_keyword_inst(self, sentence, w_star, n_similar=5):
        """
        Generates a keyword-based instruction based on a sentence.

        :param sentence: Input sentence.
        :param w_star: A word that represents a sentence.
        :param n_similar: Number of similar-sounding words to include.
        :return: Keyword instruction as a formatted string.
        """
        w_mkn_star = self.find_paralel_ind2mkn(w_star)
        similar_words = self.get_similar_sounding_words(w_mkn_star, n_similar)
        sentences = []
        for word in similar_words:
            sentences.extend(self.get_example_sentences(word, n=1))
        context = "\n".join(sentences)

        # track the number of times each prompt type is used
        instruction_template = random.choice(self.prompt_instruction)
        self.prompt_instruction_counts[instruction_template] += 1
        keyword_template = random.choice(self.keyword_prompt)
        self.keyword_prompt_counts[keyword_template] += 1

        instruction = instruction_template.format(SOURCE="Indonesia", INPUT=sentence, TARGET="Melayu Kupang")
        final_prompt = keyword_template.format(TARGET="Melayu Kupang", CONTEXT=context, INSTRUCTION=instruction)

        return final_prompt

    def generate_list_group_label_inst(self, sentence, w_stars, n_nearest=2):
        """
        Generates a list-group-label instruction based on a sentence.

        :param sentence: Input sentence.
        :param w_stars: List of words represents a sentence.
        :param n_nearest: Number of nearest word in list of words.
        :return: Keyword instruction as a formatted string.
        """
        w_parallel = []
        for idx, word in enumerate(w_stars):
            pair = self.find_paralel_ind2mkn([word])
            w_parallel.append((pair, word[0]))

        groups = []
        for idx, word in enumerate(w_parallel):
            try:
                nearest_words = self.w2v_model.most_similar(word[0], topn=n_nearest)
                nearest_words.append((word[0], 1))
            except:
                nearest_words = self.random_word_prob()[:n_nearest]
                nearest_words.append((word[0], 1))
            groups.append({f"Label {idx+1}": nearest_words})

        context = ""
        for idx, group in enumerate(groups):
            for key, value in group.items():
                context += f"{key}:"
                for word in value:
                    context += f" {word[0]},"
                context += "\n"

        # track the number of times each prompt type is used
        instruction_template = random.choice(self.prompt_instruction)
        self.prompt_instruction_counts[instruction_template] += 1
        list_group_label_template = random.choice(self.list_group_label_prompt)
        self.list_group_label_prompt_counts[list_group_label_template] += 1

        instruction = instruction_template.format(SOURCE="Indonesia", INPUT=sentence, TARGET="Melayu Kupang")
        final_prompt = list_group_label_template.format(SOURCE="Indonesia", TARGET="Melayu Kupang", CONTEXT=context, INSTRUCTION=instruction)

        return final_prompt

def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("Starting the process...")

    # Step 0: Ensure cache directory
    cache_dir = f"{current_dir}/cache"
    ensure_cache_directory(cache_dir)

    # Step 1: Load resources
    output_vec_zip_path, _ = download_and_load_fasttext(logger)
    w2v = load_word_embeddings(logger, output_vec_zip_path)
    bilingual_dict, dictionary = extract_dictionary()
    phonetic_rules = extract_phonetic_rules(dictionary)

    # Step 2: Load parallel dataset
    parallel_data = load_parallel_dataset(logger)

    # Step 3: Initialize models
    logger.info("Initializing the instruction generator...")
    generator_train = InstructionGenerator(bilingual_dict, w2v, phonetic_rules)
    generator_test = InstructionGenerator(bilingual_dict, w2v, phonetic_rules)

    # Step 4: Extract keywords
    keyword_train_cache = f"{cache_dir}/keywords_train.json"
    keyword_test_cache = f"{cache_dir}/keywords_test.json"

    keywords_train = extract_or_load_keywords(logger, parallel_data['train']["ind"], keyword_train_cache)
    keywords_test = extract_or_load_keywords(logger, parallel_data['test']["ind"], keyword_test_cache)
    
    # Step 5: Generate instructions
    train_data = generate_instructions(logger, parallel_data['train'], generator_train, keywords_train)
    test_data = generate_instructions(logger, parallel_data['test'], generator_test, keywords_test)

    # Step 6: Push dataset to Hugging Face Hub  
    push_to_hub(logger, train_data, test_data, "joanitolopo/kupangmalay-instructcorpus-v2")

    # Get prompt counts
    prompt_counts_train = generator_train.get_prompt_counts()
    prompt_counts_test = generator_test.get_prompt_counts()

    # Save prompt counts to a file
    with open("prompt_counts_train.json", "w") as f:
        json.dump(prompt_counts_train, f, indent=4)
    with open("prompt_counts_test.json", "w") as f:
        json.dump(prompt_counts_test, f, indent=4)

    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()