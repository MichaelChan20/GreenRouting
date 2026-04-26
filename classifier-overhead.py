import os
import gc
import json
import torch
import pandas as pd
import numpy as np
import random
import inspect
import time
import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria, RobertaTokenizer, RobertaModel

from datasets import load_dataset, concatenate_datasets
from evaluate import load as load_metric
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from collections import Counter




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


import random


def main(model_names, batch_number, sleep_seconds, iterations=10):
    iterations = 30
    sleep_seconds = 15
    eval_dataset = load_dataset("openai/openai_humaneval", split="test")
    mbpp_dataset = load_dataset("mbpp")
    
    mbpp_combined = concatenate_datasets([
        mbpp_dataset["train"],
        mbpp_dataset["test"],
        mbpp_dataset["validation"],
        mbpp_dataset["prompt"]
    ])
    
    all_prompts = list(mbpp_combined["text"]) + list(eval_dataset["prompt"])
    random_values = [random.random() for _ in all_prompts]
    random_ints = [random.randint(0, 1) for _ in all_prompts]
    #eval_dataset = eval_dataset.select(range(2))
    
    # models = {}
    # for model_name in model_names:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
    #     #if instruct model, mark as instruct
    #     instruct = False
    #     if "deepseek" in model_name.lower():
    #         instruct = True
            
    #     models[model_name] = (model, tokenizer, instruct)
        
    models = {}
    bert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to("cuda").eval()
    
    
    embeds = embed_texts(all_prompts, bert_tokenizer,bert_model, 100)
    for model_name in model_names:
        instruct = False
        if "lr" in model_name.lower():
            model = LogisticRegression(
                max_iter=2000,
                
            )
            model.fit(embeds, random_ints)
            
        else:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(embeds, random_values)
            
            
        models[model_name] = (model, "tokenizer", instruct)
        
        
        
        
    experiments = []
    

            
    for j in range(len(mbpp_dataset["train"])):
        example = mbpp_dataset["train"][j]
        example["prompt"] = example["text"] + " Your code should satisfy these tests:\n\n" + "\n".join(example["test_list"][:3])
        example['task_id'] = "mbpp/" + str(example['task_id'])
        task_id = example["task_id"]
        prompt = example["prompt"]
        
        for model_name, (model, tokenizer, instruct) in models.items():
            for i in range(iterations):
                experiments.append({
                    "task_id": task_id,
                    "prompt": prompt,
                    "model_name": model_name,
                    "iteration": i
                })



    samples = []
    random.seed(42)
    random.shuffle(experiments)
    
    ##warmup
    base_warmed_up = False
    instruct_warmed_up = True
    for experiment in experiments:
        task_id = experiment["task_id"]
        prompt = experiment["prompt"]
        model_name = experiment["model_name"]
        iteration = experiment["iteration"]
        instruct = models[model_name][2]
        
        model, tokenizer, instruct = models[model_name]
        #repeats = 200
        #prompts = np.repeat(prompt, repeats).tolist()
        prompts = all_prompts
        
        
        #Warmup GPU on each model with 50 generations
        if instruct == False and not base_warmed_up:
            for _ in range(50):
                #completion = generate_Qwen(experiment, tokenizer, model)
                #embed_texts(prompts, tokenizer, model, 1)
                for i in range(len(embeds)):
                    model.predict([embeds[i]])
            base_warmed_up = True
        elif instruct == True and not instruct_warmed_up:
            for _ in range(50):
                #completion = generate_one(experiment, tokenizer, model)
                model.predict([embeds[0]])
            instruct_warmed_up = True
            
        if base_warmed_up and instruct_warmed_up:
            time.sleep(60) # Sleep a minute after intensive warmup
            break
    
    for experiment in experiments:
        task_id = experiment["task_id"]
        prompt = experiment["prompt"]
        model_name = experiment["model_name"]
        iteration = experiment["iteration"]        
        model, tokenizer, instruct = models[model_name]
        #repeats = 200
        #prompts = np.repeat(prompt, repeats).tolist()
        prompts = all_prompts
        
        time.sleep(sleep_seconds)
        start_time = time.time()
        if instruct == False:
            #completion = generate_Qwen(experiment, tokenizer, model)
            for i in range(len(embeds)):
                    model.predict(embeds[i])
        else:
            completion = generate_one(experiment, tokenizer, model)
        end_time = time.time()
        samples.append({
            "task_id": task_id,
            "prompt": prompt,
            "model_name": model_name,
            "iteration": iteration,
            #"completion": completion,
            #"repeats": repeats,
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
    
    #model_names = ["deepseek-ai/deepseek-coder-1.3b-instruct"]#, "deepseek-ai/deepseek-coder-6.7b-instruct"]
    #model_names = ["deepseek-ai/deepseek-coder-1.3b-instruct"]
    #model_names = ["Qwen/Qwen2.5-Coder-3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"]
    #model_names = ["microsoft/codebert-base"]
    model_names = ["lr", "xgb"]
    main(model_names, args.batch, args.sleep)