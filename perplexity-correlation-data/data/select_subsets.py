#conda_env: perplexity
from datasets import load_dataset
import random
import json 
import os

def count_tokens(data_path, threshold = 0.98):
    files = []
    token_count = 0

    for file in os.listdir(data_path): 
        files.append(
            [file, int(file.split(".")[0])])
    files.sort(key=lambda x: x[1]) 
    for file in files:
        print(f"On file {file}")
        with open(f"{data_path + file[0]}", "r") as file: 
            for line in file: 
                data = json.loads(line)
                if data["probability"] > threshold and "1" in data["label"]:
                    token_count += data["token_length"]
        print("Current token count: ", token_count)
    return token_count

def preselect_dataset(data_path, save_path, threshold=0.98):
    files = []
    for file in os.listdir(data_path): 
        files.append(
            [file, int(file.split(".")[0])])
    files.sort(key=lambda x: x[1]) 
    with open(save_path, "w") as json_file: 
        for file in files:
            print(f"On file {file}")
            with open(f"{data_path + file[0]}", "r") as f:
                index = - 1
                for index, line in enumerate(f):
                    data = json.loads(line)
                    if data["probability"] > threshold and "1" in data["label"]:
                        data["file_num"] = file[1] 
                        data["index"] = index
                        json.dump(data, json_file)
                        json_file.write("\n")


def random_subset(data_path, save_path, num_token=3_400_000_000):
    files = []
    for file in os.listdir(data_path): 
        files.append(
            [file, int(file.split(".")[0])])
    files.sort(key=lambda x: x[1]) 
    num_tokens = 0
    with open(save_path, "w") as json_file: 
        while num_tokens < num_token:
            current_file = random.choice(files)
            print(f"On file {current_file}")
            print(f"Current token count: {num_tokens}")
            with open(f"{data_path + current_file[0]}", "r") as f:
                num_lines = sum(1 for _ in f)
                print(f"Number of lines in file: {num_lines}")
                f.seek(0)  
                index = -1
                random_numbers = random.sample(range(num_lines), 1000)
                for index, line in enumerate(f):
                    data = json.loads(line)
                    if index in random_numbers: 
                        num_tokens += data["token_length"]
                        data["file_num"] = current_file[1] 
                        data["index"] = index
                        json.dump(data, json_file)
                        json_file.write("\n")

if __name__ == "__main__":
    path = "/home/allanz/perplexity-correlation-data/data/reformatted_data/"
    #preselect_dataset(path, "/home/allanz/perplexity-correlation-data/data/selected_subsets/preselect.jsonl", threshold=0.97)
    random_subset(path, "/home/allanz/perplexity-correlation-data/data/selected_subsets/random.jsonl", num_token=3_200_000_000)

