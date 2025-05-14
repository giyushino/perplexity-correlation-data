from datasets import load_dataset 
import json
import os
from collections import defaultdict 
import boto3 
import random
import concurrent

# aws s3 cp s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_0_of_10/shard_00001354_processed.jsonl.zstd ./shard_00001354_processed.jsonl.zst 

# We can a sample pool of 30B tokens
# every global shard has 10 local each local shard has ~3400 jsonl files
# each shard has about ~50 million tokens, we would need 600 shards 
# 10 global -> 10 local -> 3400 jsonl, pick 6 jsonl from each 

s3 = boto3.client("s3")
save_path = "/home/allanz/perplexity-correlation/data/raw_data/"
bucket = "commoncrawl"
os.makedirs(save_path, exist_ok=True)

def download_raw_data(files_per_shard, save_path=save_path):
    count = 0
    for i in range(1, 11):
        for j in range(10):
            global_num = f"{i:02d}"
            local_num = j
            shard_nums = [f"{x:04d}" for x in random.sample(range(3400), files_per_shard)]
            
            for shard_num in shard_nums:
                key = f"contrib/datacomp/DCLM-refinedweb/global-shard_{global_num}_of_10/local-shard_{local_num}_of_10/shard_0000{shard_num}_processed.jsonl.zstd"
                name = f"global{global_num}_local{local_num}_shard_0000{shard_num}_processed.jsonl.zst"
                try:  
                    s3.download_file(bucket, key, os.path.join(save_path, name))
                    count += 1
                except Exception as e:
                    print(f"\n[ERROR] Failed on {name}: {e}")
                print(f"On file {count} out of {10 * 10 * files_per_shard}", end = "\r")

def download_concurrently():
    s3 = boto3.client("s3")
    save_path = "/home/allanz/perplexity-correlation-data/data/raw_data/"
    bucket = "commoncrawl"
    os.makedirs(save_path, exist_ok=True)

    def download_file(key, name, save_path):
        try:
            s3.download_file(bucket, key, os.path.join(save_path, name))
            print(f"Downloaded {name}")
        except Exception as e:
            print(f"\n[ERROR] Failed on {name}: {e}")

    def download_raw_data(files_per_shard, save_path=save_path):
        count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor to download concurrently
            for i in range(1, 11):
                for j in range(10):
                    global_num = f"{i:02d}"
                    local_num = j
                    shard_nums = [f"{x:04d}" for x in random.sample(range(3400), files_per_shard)]
                    
                    # Submit download tasks concurrently
                    for shard_num in shard_nums:
                        key = f"contrib/datacomp/DCLM-refinedweb/global-shard_{global_num}_of_10/local-shard_{local_num}_of_10/shard_0000{shard_num}_processed.jsonl.zstd"
                        name = f"global{global_num}_local{local_num}_shard_0000{shard_num}_processed.jsonl.zst"
                        executor.submit(download_file, key, name, save_path)  # Asynchronously submit the download task
                        count += 1
                        print(f"On file {count} out of {10 * 10 * files_per_shard}", end="\r")


def compute_domains(dataset_path, save_path):
    """
    Computes the domains from a dataset and saves them to a file
    Args:
        dataset_path (str): Path to the dataset file
        save_path (str): Path to save the domain information

    Returns:
        None
    """
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
    # Example usage
    #compute_domains("/home/allanz/perplexity-correlation-data/data/selected_subsets/random.jsonl", "/home/allanz/perplexity-correlation-data/data/urls/random.jsonl")



def compile_urls(data_path):
    files = []
       
    for file in os.listdir(data_path): 
        files.append(
            [file, int(file.split(".")[0])])
    files.sort(key=lambda x: x[1]) 
    domain_map = defaultdict(lambda: defaultdict(list))    
    for filename in files:
        print(f"on {filename[0]}")
        with open(f"{data_path + filename[0]}", 'r') as f:
            for line in f: 
                data = json.loads(line)
                domain = data['domain']
                domain_map[domain][filename[0]] = data['indices']
                
    with open("/home/allanz/perplexity-correlation-data/data/urls/all.jsonl", "w") as f:
        for domain, files in domain_map.items():
            entry = {
            "domain": domain,
            "files": files  
            }
            f.write(json.dumps(entry) + "\n")

    # Example usage
    #data_path = "/home/allanz/perplexity-correlation-data/data/domain_stats/"
    #urls = "/home/allanz/perplexity-correlation-data/data/urls/"
    #compile_urls(data_path)

