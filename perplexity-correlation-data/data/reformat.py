#conda_env: perplexity

from transformers import AutoTokenizer
from collections import defaultdict
import zstandard as zstd
import fasttext
import argparse
import json 
import os
import re

BASE_PATH = os.environ["WORK"]

def preselect_classifier():
    return fasttext.load_model("/home/allanz/.cache/huggingface/hub/models--hkust-nlp--preselect-fasttext-classifier/snapshots/467086fdc507d674dbb041971f1e9cae3c6e2140/PreSelect-classifier.bin")

def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def extract_zsl(file_path, file_num, fasttext_classifier, tokenizer):
    """
    Extract data from the compressed .jsonl.zst file
    """
    total_tokens = 0
    domain_count = defaultdict(int)
    domain_indices = defaultdict(list)
    entry_index = 0

    print(f"Working on file {file_path}")
    new_json = os.path.join(BASE_PATH, "perplexity-correlation-data", "data", "reformatted_data", f"{file_num}.jsonl")
    with open(new_json, "w") as j:
        with open(file_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                for i, raw_line in enumerate(reader.read().splitlines(), start=1):
                    try:
                        line = raw_line.decode('utf-8')
                        json_line = json.loads(line)
                        text = json_line["text"]
                        cleaned_text = text.replace("\n", " ")
                        token_length = len(tokenizer(cleaned_text, return_tensors="pt")["input_ids"][0])
                        total_tokens += token_length
                        domain = json_line["url"].split("//")[1].split("/")[0]

                        domain_count[domain] += 1
                        domain_indices[domain].append(entry_index)

                        data = {
                            "text": cleaned_text,
                            "label": label[0],
                            "probability": probability[0],
                            "token_length": token_length,
                            "url": domain
                        }
                        json.dump(data, j)
                        j.write("\n")
                        entry_index += 1

                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
                    print(f"Entry {i} || tokens {token_length}", end="\r")

    domain_stats_path = os.path.join(BASE_PATH, "perplexity-correlation-data", "data", "domain_stats", f"{file_num}.jsonl")
    os.makedirs(os.path.dirname(domain_stats_path), exist_ok=True)
    with open(domain_stats_path, "w") as file:
        for domain in domain_count:
            json.dump({
                "domain": domain,
                "count": domain_count[domain],
                "indices": domain_indices[domain]
            }, file)
            file.write("\n")

    with open(f"/home/allanz/perplexity-correlation-data/data/tokens/token_count{file_num}.txt", "w") as file: 
        file.write(f"{total_tokens}")

def extract_sort_key(filename):
    """
    sort files through regex
    """
    match = re.search(r"global(\d+)_local(\d+)_shard_(\d+)_", filename)
    if match:
        return tuple(map(int, match.groups()))
    else:
        return (float('inf'), float('inf'), float('inf'))  




def extract_data(start_index, end_index, data_path="/home/allanz/perplexity-correlation-data/data/raw_data/"):
    """
    Extract da data
    """
    start_index = int(start_index)
    end_index = int(end_index)
    fasttext_classifier = preselect_classifier()
    tokenizer = get_tokenizer() 
    
    files = [file for file in os.listdir(data_path)]
    files.sort(key=extract_sort_key)

    for i in range(start_index, end_index):
        file_path = f"{data_path + files[i]}"
        extract_zsl(file_path, i, fasttext_classifier, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from compressed .jsonl.zst files.")
    parser.add_argument("--start_index", type=str, default=0, help="Start index for processing.")
    parser.add_argument("--end_index", type=str, default=599, help="End index for processing.")
    args = parser.parse_args()
    extract_data(args.start_index, args.end_index)
