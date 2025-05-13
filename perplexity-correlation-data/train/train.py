#conda_env: perplexity
import os
import wandb
from datasets import load_dataset
from transformers import (Trainer, TrainingArguments,
                          GPTNeoXForCausalLM, AutoConfig, 
                          AutoTokenizer
                          )

def tokenize_dataset(example):
    tokens = tokenizer(example["text"], truncation=True, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def train(model_size = "EleutherAI/pythia-160m-deduped"):
    #dataset = load_dataset("json", data_files = dataset_path) 
    model = GPTNeoXForCausalLM.from_pretrained(model_size)
    wandb.init(project="perplexity-llm-pretraining")
    # Reinitialize model to have random weights, training from scratch 
    config = AutoConfig.from_pretrained(model_size)
    reinit_model = GPTNeoXForCausalLM(config)
    # Hyperparameters from paper (Appendix G)
    training_args = TrainingArguments(
        output_dir= os.path.join(base_path, "perplexity-correlation-data-testing", "data", "model_weights"), 
        per_device_train_batch_size=128, 
        learning_rate = 5e-3, 
        warmup_ratio=0.1,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        gradient_accumulation_steps=1,
        report_to="wandb",   
        #ddp_backend="nccl",    
    )
    
    trainer = Trainer(
        model = reinit_model, 
        args = training_args, 
        dataset = dataset  
    )

    return 0


if __name__ == "__main__":
    """
    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
    )
    """
    #dataset = load_dataset("json", data_files="/home/allanz/perplexity-correlation-data/data/selected_subsets/preselect.jsonl", split="train")
    #tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
    #print(tokenized_dataset)
