from datasets import load_dataset 


preselect_dataset = load_dataset("json", data_files="/home/allanz/perplexity-correlation-data/data/selected_subsets/preselect.jsonl", split="train")
for i in range(10):
    print(preselect_dataset[i])