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

# os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from collections import Counter
class RepeatedLinesStopping(StoppingCriteria):
    def __init__(self, tokenizer, window_lines=5, repeat_threshold=3):
        self.tokenizer = tokenizer
        self.window_lines = window_lines
        self.repeat_threshold = repeat_threshold

    def __call__(self, input_ids, scores, **kwargs):

        text = self.tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        )


        lines = [l.strip() for l in text.splitlines() if l.strip()]

        if len(lines) < self.window_lines:
            return False

        last_lines = lines[-self.window_lines:]
        counts = Counter(last_lines)

        return any(c >= self.repeat_threshold for c in counts.values())
    
class EarlyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.opened_bracket = False

    # Detect closing ``` and stop early
    def __call__(self, input_ids, scores, **kwargs):
        if (input_ids[0][-1] == self.tokenizer.convert_tokens_to_ids("```")):
            if self.opened_bracket:
                return True
            else:
                self.opened_bracket = True
        return False

class AssertFilter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Expandable with maybe other common excessive generation signs
        self.patterns = [
            tokenizer.encode("assert", add_special_tokens=False),
            tokenizer.encode("# Test cases", add_special_tokens=False),
        ]
        
        self.max_len = max(len(p) for p in self.patterns) #Highest pattern length to decide window length
        

        self.replacement_id = tokenizer.convert_tokens_to_ids("```")
        
    # Check for assert generation, if generated replace with closing bracket and stop
    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]
        seq_len = seq.shape[0]

        if seq_len < self.max_len:
            return False

        tail = seq[-self.max_len:]

        for pat in self.patterns:
            pat_len = len(pat)
            pat_tensor = torch.tensor(pat, device=seq.device)

            if torch.all(tail[-pat_len:] == pat_tensor):

                seq[-pat_len] = self.replacement_id
                return True

        return False

def create_stopping_criteria(tokenizer):
    return StoppingCriteriaList([
        RepeatedLinesStopping(
            tokenizer,
            window_lines=5,
            repeat_threshold=3
        ),
        AssertFilter(tokenizer),
        EarlyStoppingCriteria(tokenizer)
    ])


##
def build_instruction(languge: str, question: str):
    return '''
Please only respond with a codeblock and do not explain anything. Complete the following task:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())


def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]
    print(func_lines)
    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_generation_code(example: str, verbose: bool=False):
    #task_id = example['task_id']
    output = example.get('output', example.get("gpt_completion"))
    question = example["prompt"].strip()
    setting = {
        'full_name': 'Python',
        'indent': 2,
    }
    lang = setting['full_name']
    indent = setting['indent']
    #print(output)
    try:
        code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        if verbose:
            print(">>> Task: {}\n{}".format(task_id, code_block))
        
        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]
        
        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent-1] == ' ':
                indent += 1
            
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)

        body = code_block[start:end]

    
        generation = func_prefix + '\n' + body + '\n'
        example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        example['generation'] = example['prompt'] + '\n' + output
    
    return example

def extract_codeblock(text: str) -> str:
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def generate_one(example, tokenizer, model):
    prompt = build_instruction('Python', example['prompt'])
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"
    
    stopping_criteria = create_stopping_criteria(tokenizer)

    outputs = model.generate(
        inputs, 
        max_new_tokens=1024,
        do_sample=False,
        # top_p=0.95,
        # temperature=temperature,
        pad_token_id=stop_id,
        eos_token_id=stop_id,
        #max_length=2048
        stopping_criteria=stopping_criteria,
    )

    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    #example['output'] = output

    #print(extract_codeblock(output))
    return output

def generate_Qwen(example, tokenizer, model):
    prompt = build_instruction('Python', example['prompt'])
    messages = [
        {"role": "system", "content": "You are an intelligent programming assistant to produce Python algorithmic solutions and only respond with code without explanations."},
        {"role": "user", "content": prompt}
    ]

    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"
    
    stopping_criteria = create_stopping_criteria(tokenizer)

    generated_ids = model.generate(
        inputs, 
        #max_new_tokens=1024,
        do_sample=False,
        # top_p=0.95,
        # temperature=temperature,
        stopping_criteria=stopping_criteria,
        pad_token_id=stop_id,
        eos_token_id=stop_id,
        max_new_tokens=1024,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs, generated_ids)
    ]


    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #example['output'] = output

    #print(extract_codeblock(output))
    return output


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
    for model_name in model_names:
        #tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        #model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).to("cuda").eval()
        #if instruct model, mark as instruct
        instruct = False
        if "deepseek" in model_name.lower():
            instruct = True
            
        models[model_name] = (model, tokenizer, instruct)
        
        
        
    experiments = []
    
    # for example in eval_dataset:
    #     task_id = example["task_id"]
    #     prompt = example["prompt"]
        
    #     for model_name, (model, tokenizer, instruct) in models.items():
    #         for i in range(iterations):
    #             experiments.append({
    #                 "task_id": task_id,
    #                 "prompt": prompt,
    #                 "model_name": model_name,
    #                 "iteration": i
    #             })

    # for j in range(len(mbpp_dataset["test"])):
    #     example = mbpp_dataset["test"][j]
    #     example["prompt"] = example["text"]# + " Your code should satisfy these tests:\n\n" + "\n".join(example["test_list"][:3])
    #     example['task_id'] = "mbpp/" + str(example['task_id'])
    #     task_id = example["task_id"]
    #     prompt = example["prompt"]
        
    #     for model_name, (model, tokenizer, instruct) in models.items():
    #         for i in range(iterations):
    #             experiments.append({
    #                 "task_id": task_id,
    #                 "prompt": prompt,
    #                 "model_name": model_name,
    #                 "iteration": i
    #             })
                
    # for j in range(len(mbpp_dataset["train"])):
    #     example = mbpp_dataset["train"][j]
    #     example["prompt"] = example["text"]# + " Your code should satisfy these tests:\n\n" + "\n".join(example["test_list"][:3])
    #     example['task_id'] = "mbpp/" + str(example['task_id'])
    #     task_id = example["task_id"]
    #     prompt = example["prompt"]
        
    #     for model_name, (model, tokenizer, instruct) in models.items():
    #         for i in range(iterations):
    #             experiments.append({
    #                 "task_id": task_id,
    #                 "prompt": prompt,
    #                 "model_name": model_name,
    #                 "iteration": i
    #             })
    
    # for j in range(len(mbpp_dataset["validation"])):
    #     example = mbpp_dataset["validation"][j]
    #     example["prompt"] = example["text"]# + " Your code should satisfy these tests:\n\n" + "\n".join(example["test_list"][:3])
    #     example['task_id'] = "mbpp/" + str(example['task_id'])
    #     task_id = example["task_id"]
    #     prompt = example["prompt"]
        
    #     for model_name, (model, tokenizer, instruct) in models.items():
    #         for i in range(iterations):
    #             experiments.append({
    #                 "task_id": task_id,
    #                 "prompt": prompt,
    #                 "model_name": model_name,
    #                 "iteration": i
    #             })
                
    # for j in range(len(mbpp_dataset["prompt"])):
    #     example = mbpp_dataset["prompt"][j]
    #     example["prompt"] = example["text"]# + " Your code should satisfy these tests:\n\n" + "\n".join(example["test_list"][:3])
    #     example['task_id'] = "mbpp/" + str(example['task_id'])
    #     task_id = example["task_id"]
    #     prompt = example["prompt"]
        
    #     for model_name, (model, tokenizer, instruct) in models.items():
    #         for i in range(iterations):
    #             experiments.append({
    #                 "task_id": task_id,
    #                 "prompt": prompt,
    #                 "model_name": model_name,
    #                 "iteration": i
    #             })
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
                embed_texts(prompts, tokenizer, model, 1)
            base_warmed_up = True
        elif instruct == True and not instruct_warmed_up:
            for _ in range(50):
                completion = generate_one(experiment, tokenizer, model)
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
            embed_texts(prompts, tokenizer, model, 1)
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
    model_names = ["microsoft/codebert-base"]
    
    main(model_names, args.batch, args.sleep)