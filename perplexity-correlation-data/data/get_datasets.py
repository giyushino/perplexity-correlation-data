from datasets import load_dataset 
import json
from collections import defaultdict 

def compute_domains(dataset_path, save_path):
    domains = set() 
    domain_count = defaultdict(int)  
    domain_indices = defaultdict(list)  
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    for index, entry in enumerate(dataset): 
        cleaned_url = entry["url"]
        
        domain_count[cleaned_url] += 1
        domain_indices[cleaned_url].append(index)
        
        if cleaned_url not in domains: 
            domains.add(cleaned_url)

    with open(save_path, "w") as file: 
        for domain in domain_count:
            print(f"Domain: {domain}, Count: {domain_count[domain]}, Indices: {domain_indices[domain]}")
            json.dump({"domain": domain, "indices": domain_indices[domain], "count": domain_count[domain]}, file) 
            file.write("\n")
    return 0

print("random")
compute_domains("/home/allanz/perplexity-correlation-data/data/selected_subsets/random.jsonl", "/home/allanz/perplexity-correlation-data/data/urls/random.jsonl")