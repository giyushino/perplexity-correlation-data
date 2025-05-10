import os
from datasets import load_dataset
from transformers import (Trainer, TrainingArguments,
                          GPTNeoXForCausalLM, AutoConfig
                          )

base_path = os.environ["WORK"]

def train(dataset_path, model_size = "EleutherAI/pythia-160m-deduped"):
    dataset = load_dataset("json", dataset_path) 
    model = GPTNeoXForCausalLM.from_pretrained(model_size)

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
    train()
