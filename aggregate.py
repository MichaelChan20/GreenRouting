import pandas as pd

from collections import defaultdict
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer

from datasets import load_dataset

import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from matplotlib.lines import Line2D
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix


def collect_files(folders):
    json_files = []
    csv_files = []

    for folder in folders:
        folder = Path(folder)

        json_files.extend(sorted(folder.glob("*.json")))
        csv_files.extend(sorted(folder.glob("*.csv")))

    if len(json_files) != len(csv_files):
        raise ValueError(
            f"Mismatch: {len(json_files)} JSON vs {len(csv_files)} CSV files"
        )

    return json_files, csv_files

def load_tasks(json_files):
    
    all_tasks = []

    for path in json_files:
        with open(path, "r") as f:
            tasks = json.load(f)
            all_tasks.extend(tasks)

    return all_tasks


def load_df(json_files, csv_files):
    all_tasks = []
    all_dfs = []

    for json_path, csv_path in zip(json_files, csv_files):


        with open(json_path, "r") as f:
            tasks = json.load(f)
            all_tasks.extend(tasks)


        df = pd.read_csv(csv_path)

        df["Time"] = pd.to_numeric(df["Time"], errors="coerce") / 1000
        df = df.dropna(subset=["Time"])

        df = df.set_index("Time").sort_index()

        df["GPU0_POWER (mWatts)"] = pd.to_numeric(
            df["GPU0_POWER (mWatts)"], errors="coerce"
        )

        df["GPU0_POWER (mWatts)"] = df["GPU0_POWER (mWatts)"].interpolate(method="linear")

        all_dfs.append(df)


    merged_df = pd.concat(all_dfs).sort_index()

    return merged_df, all_tasks

def compute_energy_trapezoid(df, start, end, power_col="GPU0_POWER (mWatts)"):

    df_interval = df.loc[start:end].copy()
    
    if df_interval.empty:
        return None
    
    times = df_interval.index.to_numpy()
    powers_watts = df_interval[power_col].to_numpy() / 1000  # mW to W
    
    if len(times) == 1:
        duration = end - start
        return powers_watts[0] * duration
    
    # Interpolate start
    if start < times[0]:
        start_power = np.interp(start, [times[0], times[1]], [powers_watts[0], powers_watts[1]])
        times = np.insert(times, 0, start)
        powers_watts = np.insert(powers_watts, 0, start_power)
    
    # Interpolate end
    if end > times[-1]:
        end_power = np.interp(end, [times[-2], times[-1]], [powers_watts[-2], powers_watts[-1]])
        times = np.append(times, end)
        powers_watts = np.append(powers_watts, end_power)
    
    energy_joules = np.trapezoid(powers_watts, x=times)
    return energy_joules


#This is just for ordering, HE priority
def extract_task_num(task_id):
    match = re.search(r'HumanEval/(\d+)', task_id)
    if not match:
        match = re.search(r'mbpp/(\d+)', task_id)
        return  1000 + int(match.group(1)) if match else float('inf')
    return int(match.group(1)) if match else float('inf')


def aggregate(tasks, df):
    results = defaultdict(lambda: {
        "task_id": None,
        "model_name": None,
        "measurements": []
    })
    
    HE_dataset = load_dataset("openai/openai_humaneval", split="test")
    MBPP_dataset = load_dataset("mbpp")

    task_to_prompt = {ex["task_id"]: ex["prompt"] for ex in HE_dataset}
    task_to_prompt.update({"mbpp/" + str(ex["task_id"]): ex["text"] for ex in MBPP_dataset["test"]})
    task_to_prompt.update({"mbpp/" + str(ex["task_id"]): ex["text"] for ex in MBPP_dataset["train"]})
    task_to_prompt.update({"mbpp/" + str(ex["task_id"]): ex["text"] for ex in MBPP_dataset["validation"]})
    task_to_prompt.update({"mbpp/" + str(ex["task_id"]): ex["text"] for ex in MBPP_dataset["prompt"]})


    tasks_sorted = sorted(tasks, key=lambda x: extract_task_num(x['task_id']))

    for task in tasks_sorted:
        key = (task["task_id"], task["model_name"])
        start = float(task["start_time"])
        end = float(task["end_time"])
        duration = end - start


        #Calculate GPU energy using trapezoidal rule

        interval_trapz = df.loc[start:end, "GPU0_POWER (mWatts)"]
        times = interval_trapz.index.to_numpy()
        powers_watts = interval_trapz.to_numpy() / 1000
        energy_trapz = compute_energy_trapezoid(df, start, end)
        
        
        
        interval = df.loc[start:end]
        avg_power_mw = interval["GPU0_POWER (mWatts)"].mean() if not interval.empty else None
        #print(avg_power_mw)
        energy_joules = (avg_power_mw / 1000) * duration if avg_power_mw is not None else None
        results[key]["task_id"] = task["task_id"]
        results[key]["model_name"] = task["model_name"]
        results[key]["completion"] = task["completion"]
        results[key]["measurements"].append({
            "iteration": task["iteration"],
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(duration, 3),
            "avg_gpu_power_watts": round(avg_power_mw / 1000, 3) if avg_power_mw is not None else None,
            #"energy_joules": round(energy_joules, 3) if energy_joules is not None else None,
            "energy_joules": round(energy_trapz, 3) if energy_trapz is not None else None
        })


    grouped_results = list(results.values())


    tokenizers = {}
    def get_tokenizer_for_model(model_name):
        if model_name not in tokenizers:
            if "deepseek" in model_name.lower():
                # DeepSeek Coder tokenizer
                tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            elif "qwen" in model_name.lower():
                # Qwen-Coder tokenizer
                tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            else:
                # fallback or raise
                raise ValueError(f"Unknown tokenizer for model: {model_name}")
        return tokenizers[model_name]


    rows = []
    for group in results.values():
        durations = [m["duration"] for m in group["measurements"] if m["duration"] is not None]
        powers = [m["avg_gpu_power_watts"] for m in group["measurements"] if m["avg_gpu_power_watts"] is not None]
        energies = [m["energy_joules"] for m in group["measurements"] if m["energy_joules"] is not None]

        avg_duration = sum(durations) / len(durations) if durations else 0
        var_duration = pd.Series(durations).var(ddof=1) if len(durations) > 1 else 0

        avg_power = sum(powers) / len(powers) if powers else 0
        var_power = pd.Series(powers).var(ddof=1) if len(powers) > 1 else 0

        avg_energy = sum(energies) / len(energies) if energies else 0
        var_energy = pd.Series(energies).var(ddof=1) if len(energies) > 1 else 0
        
        completion = group.get("completion", "N/A")
        
        task_id = group["task_id"]
        model_name = group["model_name"]
        
        prompt = task_to_prompt.get(task_id, "Prompt not found")
        
        tokenizer = get_tokenizer_for_model(model_name)
        
        completion_tokens = tokenizer(completion, return_tensors="pt", truncation=True, max_length=1024)
        token_count = completion_tokens.input_ids.shape[1]
        
        stat, p_value = shapiro(energies)

        # print(f"\nTask: {group['task_id']} | Model: {model_name}")
        # print(f"  Avg Duration: {avg_duration:.3f} s")
        # print(f"  Duration Variance: {var_duration:.10f} s^2")
        # print(f"  Duration_std: {var_duration**0.5:.3f} s")
        # print(f"  Avg GPU Power: {avg_power:.3f} W")
        # print(f"  Power Variance: {var_power:.5f} W^2")
        # print(f"  Avg Energy: {avg_energy:.3f} J")
        # print(f"  Energy Variance: {var_energy:.5f} J^2")
        # print(f"  Energy_std: {var_energy**0.5:.3f} J")
        # print(f"  Energy_std%: {(var_energy**0.5 / avg_energy * 100) if avg_energy > 0 else 0:.2f}%")
        # print(f"  Completion: {completion}")
        # print(f"  Prompt: {prompt}\n")
        # print(f"  Token Count: {token_count} tokens")
        # print(f"Shapiro-Wilk statistic: {stat:.4f}")
        # print(f"p-value: {p_value:.4f}")
        
        rows.append({
            "task_id": task_id,
            "model": model_name,
            "prompt": prompt,
            "avg_energy": avg_energy,
            "variance_energy": var_energy,
            "completion": completion,
            "energy_std": var_energy**0.5,
            "energy_std%": (var_energy**0.5 / avg_energy * 100) if avg_energy > 0 else 0,
            "duration": avg_duration,
            "token_count": token_count,
            "p-value": p_value,
            "energy_per_token": (avg_energy / token_count) if token_count > 0 else 0
        })
    df = pd.DataFrame(rows)

    df.to_json("aggregated_data.json", orient="records", indent=2)
    return df


def plots(df, rows):
    models = list(set(row["model"] for row in rows))
    colors = ["blue", "orange"] 


    plt.figure(figsize=(10,6))

 
    parts = plt.violinplot(
        [df[df["model"] == model]["energy_per_token"] for model in models],
        showmeans=False, showmedians=False, showextrema=False
    )

  
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)


    plt.boxplot(
        [df[df["model"] == model]["energy_per_token"] for model in models],
        tick_labels=models,
        patch_artist=True,
        boxprops=dict(facecolor='none', color='black'),
        medianprops=dict(color='red')
    )

    plt.title("Energy per Token Distribution by Model")
    plt.ylabel("Joules per Token")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("./images/energy_per_token.pdf", format="pdf")
    #plt.show()

    for model in models:
        model_data = df[df["model"] == model]["avg_energy"]
        mean_energy = model_data.mean()
        
        model_data = df[df["model"] == model]["energy_per_token"]
        mean_val = model_data.mean()
        median_val = model_data.median()
        model_data = df[df["model"] == model]["token_count"]
        token_mean = model_data.mean()
        model_data = df[df["model"] == model]["completion"]
        character_mean = model_data.apply(len).mean()
        print(f"Model: {model} | Mean Tokens: {token_mean:.2f} tokens")
        print(f"Model: {model} | Mean Energy/Token: {mean_val:.4f} J | Median Energy/Token: {median_val:.4f} J")
        print(f"Model: {model} | Mean Characters: {character_mean:.2f} characters\n")
        print(f"Model: {model} | Mean Energy: {mean_energy:.3f} J\n")
        
        
        

    num_below_threshold = (df["p-value"] < 0.05).sum()
    print(f"Number of tasks with p < 0.05: {num_below_threshold}")


    #Plot P values and energy
    plt.figure(figsize=(10,6))
    plt.scatter(df["p-value"], df["avg_energy"], alpha=0.7)
    plt.axvline(0.05, color='red', linestyle='--', label='P = 0.05')
    plt.title("P-value vs Average Energy")
    plt.xlabel("P-value")
    plt.ylabel("Average Energy (Joules)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("./images/p_value_vs_energy.pdf", format="pdf")
    #plt.show()

    print(f"Total number of tasks: {len(df)}")



    plt.figure(figsize=(7,5))


    models = list(set(row["model"] for row in rows))
    colors = ["blue", "orange"] 

    for model, color in zip(models, colors):
        xs = [r["token_count"] for r in rows if r["model"] == model]
        ys = [r["avg_energy"] for r in rows if r["model"] == model]

        plt.errorbar(
            xs,
            ys,
            fmt='o',
            capsize=4,
            color=color,
            label=model
        )

    plt.xlabel("Output Token Length")
    plt.ylabel("Avg Energy (J)")
    plt.title("Energy vs Output Token Length by Model")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("./images/EnergyVsTokenLength.pdf", format="pdf")
    #plt.show()


    tokenizers = {}

    def get_tokenizer_for_model(model_name):
        if model_name not in tokenizers:
            if "deepseek" in model_name.lower():

                tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            elif "qwen" in model_name.lower():

                tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            else:

                raise ValueError(f"Unknown tokenizer for model: {model_name}")
        return tokenizers[model_name]

    # Add token counts
    for row in rows:
        model_name = row["model"]
        completion_text = row["completion"]
        
        tokenizer = get_tokenizer_for_model(model_name)
        
        # Encode completion to get token count
        tokens = tokenizer.encode(completion_text, add_special_tokens=False)
        row["output_tokens"] = len(tokens)
        


    plt.figure(figsize=(7,5))


    models = list(set(row["model"] for row in rows))
    colors = ["blue", "orange"] 

    for model, color in zip(models, colors):
        xs = [r["avg_energy"] for r in rows if r["model"] == model]
        ys = [r["energy_std%"] for r in rows if r["model"] == model]
        
        plt.errorbar(
            xs,
            ys,
            fmt='o',
            capsize=4,
            color=color,
            label=model
        )

    plt.xlabel("Energy (J)")
    plt.ylabel("Relative SD%")
    plt.title("Relative Energy deviation within task")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("./images/EnergySD.pdf", format="pdf")
    #plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folders",
        nargs="*",
        default=["results/MBPPtrain", "results/MBPPHE"],
        help="List of folders containing JSON and CSV files"
    )

    args = parser.parse_args()

    json_files, csv_files = collect_files(args.folders)

    print("JSON files:", json_files)
    print("CSV files:", csv_files)
    
    df, tasks = load_df(json_files, csv_files)
    df = aggregate(tasks, df)
    plots(df, df.to_dict(orient="records"))
    
    print("Aggregated data saved to results/aggregated_data.json")
    print("Energy measurement plots saved to ./images/ directory.")
    
    pivot = df.pivot(index="task_id", columns="model", values="correct")
    model_a = "Qwen/Qwen2.5-Coder-3B-Instruct"
    model_b = "deepseek-ai/deepseek-coder-1.3b-instruct"
    cm = confusion_matrix(pivot[model_a], pivot[model_b])
    print("\nConfusion Matrix (Qwen vs DeepSeek):")
    cm_df = pd.DataFrame(
        cm,
        index=[f"{model_a} = 0", f"{model_a} = 1"],
        columns=[f"{model_b} = 0", f"{model_b} = 1"]
    )
    print("\nConfusion Matrix (Qwen vs DeepSeek):")
    print(cm_df)