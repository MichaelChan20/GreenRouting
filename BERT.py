import json
import torch
import pandas as pd
import numpy as np
import random
import time
import argparse
from transformers import RobertaTokenizer, RobertaModel

from datasets import load_dataset, concatenate_datasets


def embed_texts(texts, tokenizer, model, batch_size=1):
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1)
        embeds.append(pooled.cpu().numpy())
    return np.vstack(embeds)


def main(model_names, batch_number, sleep_seconds, iterations=10):
    iterations = 30
    sleep_seconds = 15
    eval_dataset = load_dataset("openai/openai_humaneval", split="test")
    mbpp_dataset = load_dataset("mbpp")
        
    models = {}
    for model_name in model_names:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).to("cuda").eval()
        instruct = False
        if "deepseek" in model_name.lower():
            instruct = True
            
        models[model_name] = (model, tokenizer, instruct)
        
        
        
    experiments = []

    mbpp_combined = concatenate_datasets([
        mbpp_dataset["train"],
        mbpp_dataset["test"],
        mbpp_dataset["validation"],
        mbpp_dataset["prompt"]
    ])
    
    all_prompts = list(mbpp_combined["text"]) + list(eval_dataset["prompt"])
    for i in range(iterations):
        experiments.append({
            "task_id": "combined",
            "prompt": "all",
            "model_name": "microsoft/codebert-base",
            "iteration": i
        })


    samples = []
    random.seed(42)
    random.shuffle(experiments)
    
    ##warmup
    base_warmed_up = False
    for experiment in experiments:
        task_id = experiment["task_id"]
        prompt = experiment["prompt"]
        model_name = experiment["model_name"]
        iteration = experiment["iteration"]
        instruct = models[model_name][2]
        
        model, tokenizer, instruct = models[model_name]

        prompts = all_prompts
        
        
        #Warmup GPU on model with 50 full dataset embeddings
        if instruct == False and not base_warmed_up:
            for _ in range(50):

                embed_texts(prompts, tokenizer, model, 1)
            base_warmed_up = True
            
        if base_warmed_up:
            time.sleep(60) # Sleep a minute after warmup
            break
    
    for experiment in experiments:
        task_id = experiment["task_id"]
        prompt = experiment["prompt"]
        model_name = experiment["model_name"]
        iteration = experiment["iteration"]        
        model, tokenizer, instruct = models[model_name]
        prompts = all_prompts
        
        time.sleep(sleep_seconds)
        start_time = time.time()

        embed_texts(prompts, tokenizer, model, 1)

        end_time = time.time()
        samples.append({
            "task_id": task_id,
            "prompt": prompt,
            "model_name": model_name,
            "iteration": iteration,
            "start_time": start_time,
            "end_time": end_time
        })

    filename = f"results/{batch_number}.json"

    with open(filename, "w") as f:
        json.dump(samples, f, indent=4)
    
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Full HF model name")
    parser.add_argument("--batch", type=str, default=1, help="Batch number/timestamp for saving results")
    parser.add_argument("--sleep", type=int, default=15, help="Sleep between generations")
    args = parser.parse_args()
    
    model_names = ["microsoft/codebert-base"]
    
    main(model_names, args.batch, args.sleep)