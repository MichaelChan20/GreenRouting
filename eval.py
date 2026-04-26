import os
import gc
import json
import torch
import pandas as pd
import inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load as load_metric

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

    return code

def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]

def generate_code(prompt, model, tokenizer, eval_dataset):

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,      
            #top_k=5,            
            #top_p=0.95,          
            #num_return_sequences=5
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
    )
        
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    suffixprediction = generated_code[len(prompt):]
    suffixprediction = cleanup_code(suffixprediction, 'python', "humaneval", False, eval_dataset)
    suffixprediction = prompt + "\n" + suffixprediction
    
    
    return suffixprediction

def save_results(samples, predictions, references, results, output_dir="results"):

    os.makedirs(output_dir, exist_ok=True)

    results_dict = {
        "samples": samples,
        "predictions": predictions,
        "references": references,
        "metrics": results
    }
    with open(os.path.join(output_dir, "humaneval_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2)

    df = pd.DataFrame({
        "task_id": [s["task_id"] for s in samples],
        "completion": [s["completion"] for s in samples],
        "reference": references,
        "prediction": [p[0] for p in predictions]
    })
    df.to_csv(os.path.join(output_dir, "humaneval_results.csv"), index=False)
    
def parse_test_string(test_str: str) -> str:
    env = {}
    exec(test_str, env)
    check_fn = env.get("check")
    src = inspect.getsource(check_fn)
    src += "\ncheck(candidate)"
    return src

def main():

    eval_dataset = load_dataset("openai/openai_humaneval", split="test")
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()


    code_eval  = load_metric("code_eval")
    samples = []
    predictions = []
    references = []

    count = 0
    for example in eval_dataset:
        task_id = example["task_id"]
        prompt = example["prompt"]
        test_code = parse_test_string(example['test'])
        completion = generate_code(prompt, model, tokenizer, eval_dataset)
        samples.append({
            "task_id": task_id,
            "completion": completion
        })
        predictions.append([completion])
        references.append(test_code)
        count += 1
        if count == 20:
            break



    results = code_eval.compute(references=references, predictions=predictions, k=[1])

    save_results(samples, predictions, references, results)
    
    print("Evaluation Metrics:", results)

if __name__ == "__main__":
    main()