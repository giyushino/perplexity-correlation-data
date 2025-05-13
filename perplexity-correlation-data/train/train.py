#conda_env: perplexity
import os
import wandb
from datasets import load_dataset
from transformers import (Trainer, TrainingArguments,
                          GPTNeoXForCausalLM, AutoConfig, 
                          AutoTokenizer, DataCollatorForLanguageModeling
                          )

def tokenize_dataset(example):
    tokens = tokenizer(example["text"], truncation=True, padding=True, max_length=2048)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def train(dataset, tokenizer, model_size = "EleutherAI/pythia-160m-deduped"):
    #dataset = load_dataset("json", data_files = dataset_path) 
    # Reinitialize model to have random weights, training from scratch 
    config = AutoConfig.from_pretrained(model_size)
    reinit_model = GPTNeoXForCausalLM(config)
    wandb.init(project="perplexity-llm-pretraining", 
               config = vars(config)) 
    # Hyperparameters from paper (Appendix G)
    training_args = TrainingArguments(
        output_dir= "/home/allanz/perplexity-correlation-data/data/model_weights/", 
        run_name="test_10k", 
        per_device_train_batch_size=8, 
        learning_rate = 5e-3, 
        warmup_ratio=0.1,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        gradient_accumulation_steps=1,
        report_to="wandb",   
        num_train_epochs=1, 
        logging_steps=10,
        ddp_backend="nccl",    
    )
    
    trainer = Trainer(
        model = reinit_model, 
        args = training_args, 
        train_dataset = dataset, 
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) 

    )
    trainer.train()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    dataset = load_dataset("json", data_files = "/home/allanz/perplexity-correlation-data/data/selected_subsets/random.jsonl", split="train")
    dataset = dataset.select(range(1000))
    dataset = dataset.map(tokenize_dataset, batched=True)
    dataset = dataset.remove_columns(["text", "url", "file_num", "index", "label" ,"probability", "token_length"])
    print(dataset)
    train(dataset, tokenizer)
