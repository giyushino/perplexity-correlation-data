
from transformers import AutoTokenizer
from collections import defaultdict
import zstandard as zstd
import fasttext
import argparse
import json 
import os
import re

def preselect_classifier():
    return fasttext.load_model("/home/allanz/.cache/huggingface/hub/models--hkust-nlp--preselect-fasttext-classifier/snapshots/467086fdc507d674dbb041971f1e9cae3c6e2140/PreSelect-classifier.bin")

def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

BASE_PATH = os.environ["WORK"]

def extract_zsl(file_path, file_num, fasttext_classifier, tokenizer):
    total_tokens = 0
    domain_count = defaultdict(int)
    domain_indices = defaultdict(list)
    print(f"Working on file {file_path}")
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            for i, raw_line in enumerate(reader.read().splitlines(), start=1):
                json_line = json.loads(raw_line.decode('utf-8'))
                line = raw_line.decode('utf-8')
                json_line = json.loads(line)
                text = json_line["text"]
                cleaned_text = text.replace("\n", " ")
                label, probability = fasttext_classifier.predict(cleaned_text)
                token_length = len(tokenizer(cleaned_text, return_tensors="pt")["input_ids"][0])
                total_tokens += token_length
                domain = json_line["url"].split("//")[1].split("/")[0]

                domain_count[domain] += 1
                domain_indices[domain].append(i)

                data = {
                    "text": cleaned_text,
                    "label": label[0],
                    "probability": probability[0],
                    "token_length": token_length,
                    "url": domain
                }
                print(data)

extract_zsl("/home/allanz/perplexity-correlation-data/data/raw_data/global01_local0_shard_00000323_processed.jsonl.zst", 0, preselect_classifier(), get_tokenizer())
