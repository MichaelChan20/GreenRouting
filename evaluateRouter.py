import pandas as pd
from collections import defaultdict
import json
import re
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from matplotlib.lines import Line2D


MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[info] Using device: {device}")
print("Loading CodeBERT model and tokenizer...")



codebert_tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
codebert_model = RobertaModel.from_pretrained(MODEL_NAME).to(device).eval()

def embed_texts(texts, batch_size=100):
    embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding prompts", disable=True):
        batch = texts[i:i+batch_size]
        inputs = codebert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = codebert_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1)
        embeds.append(pooled.cpu().numpy())
    return np.vstack(embeds)



with open("results/aggregated_data_labeled.json", "r") as f:
    loaded_data = json.load(f)
df = pd.DataFrame(loaded_data)


X = df.drop(columns=["correct"]) 
y = df["correct"]
embeds = embed_texts(X["prompt"].tolist())
groups = df["task_id"]


def select_winner(group):

    if group["correct"].sum() == 1:
        winner = group[group["correct"]].iloc[0]
    else:
        # Lowest energy if both correct/incorrect
        winner = group.loc[group["avg_energy"].idxmin()]

    return winner



def linear_interpolate(x, x0, y0, x1, y1):

    x = np.asarray(x)

    if x1 == x0:
        raise ValueError("x0 and x1 must be different to define a line.")

    # slope
    m = (y1 - y0) / (x1 - x0)

    # line
    return y0 + m * (x - x0)

def percent_reduction(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape.")

    with np.errstate(divide='ignore', invalid='ignore'):
        reduction = (a - b) / a * 100
        reduction[a == 0] = np.nan  # Handle division by zero
    return reduction


def evaluate_router_humaneval(eval_dataset, candidate_models, router_func, router, mode="router"):
    correct_count = 0
    total_energy = 0.0
    correct_weak = 0
    energy_weak = 0.0
    correct_strong = 0
    energy_strong = 0.0
    
    n = len(eval_dataset)

    for i in range(len(eval_dataset)):
        prompt = eval_dataset.iloc[i]["prompt"]
        task_id = eval_dataset.iloc[i]["task_id"]

        best_model = None
        if mode == "router":
            best_model, _ = router_func(prompt, candidate_models, router)
        elif mode == "oracle":
            best_model = eval_dataset.iloc[i]["correct_model"]
        elif mode == "strong":
            best_model = candidate_models[0]
        elif mode == "random":
            best_model = random.choice(candidate_models)
        elif mode == "weak":
            best_model = candidate_models[1]

        correct = df.loc[
            (df["task_id"] == task_id) & (df["model"] == best_model),
            "correct"
        ].values
        correct = bool(correct[0]) if len(correct) > 0 else False
        correct_count += int(correct)

        energy = df.loc[
            (df["task_id"] == task_id) & (df["model"] == best_model),
            "avg_energy"
        ].values
        energy = float(energy[0]) if len(energy) > 0 else 0.0
        total_energy += energy

        #print(f"Task {task_id} {i}/{n} | Routed to {best_model} | Correct: {correct} | Energy: {energy:.2f}J")
        
        
    for i in range(len(eval_dataset)):
        task_id = eval_dataset.iloc[i]["task_id"]
        best_model = candidate_models[0]

        correct = df.loc[
            (df["task_id"] == task_id) & (df["model"] == best_model),
            "correct"
        ].values
        correct = bool(correct[0]) if len(correct) > 0 else False
        correct_strong += int(correct)

        energy = df.loc[
            (df["task_id"] == task_id) & (df["model"] == best_model),
            "avg_energy"
        ].values
        energy = float(energy[0]) if len(energy) > 0 else 0.0
        energy_strong += energy
        
    for i in range(len(eval_dataset)):
        task_id = eval_dataset.iloc[i]["task_id"]
        best_model = candidate_models[1]

        correct = df.loc[
            (df["task_id"] == task_id) & (df["model"] == best_model),
            "correct"
        ].values
        correct = bool(correct[0]) if len(correct) > 0 else False
        correct_weak += int(correct)

        energy = df.loc[
            (df["task_id"] == task_id) & (df["model"] == best_model),
            "avg_energy"
        ].values
        energy = float(energy[0]) if len(energy) > 0 else 0.0
        energy_weak += energy

        #print(f"Task {task_id} {i}/{n} | Routed to {best_model} | Correct: {correct} | Energy: {energy:.2f}J")
    correct_avg = correct_weak + correct_strong
    energy_avg = energy_weak + energy_strong
    #print(f"Baseline: accuracy: {correct_avg/(2*n):.4f} | energy: {energy_avg/(2*n):.2f}J")
    
    #interpolated accuracy and energy
    correct_diff = correct_weak - correct_strong
    energy_diff = energy_weak - energy_strong
    
    factor = (correct_count - correct_weak)/(correct_diff)
    interp_energy = energy_weak + (factor * energy_diff)
    
    
    #print(f"Interpolated energy: {interp_energy/n:.2f}J")
    #print(f"Strong_model accuracy: {correct_strong/n:.4f} | energy: {energy_strong/n:.2f}J")
    
    return correct_count, total_energy


def fold_knnc(k=5, random_state=42, fold_accuracies=None, fold_energies=None):
    global scaler
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=random_state)
    accuracies = []
    energies = []

    for train_idx, test_idx in gkf.split(df, groups=groups):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]



        train_df = (
            df_train.groupby("task_id", group_keys=False)
            .apply(select_winner)
            .loc[:, ["task_id", "prompt", "model"]]
            .rename(columns={"model": "correct_model"})
            .reset_index(drop=True)
        )
        
        test_df = (
            df_test.groupby("task_id", group_keys=False)
            .apply(select_winner)
            .loc[:, ["task_id", "prompt", "model"]]
            .rename(columns={"model": "correct_model"})
            .reset_index(drop=True)
        )
        
        
        mbpp_train_prompts = train_df["prompt"].tolist()
        train_df["correct_model"].tolist()

        mbpp_train_embeddings = embed_texts(mbpp_train_prompts)
        train_df["correct_model_encoded"] = train_df["correct_model"].astype('category').cat.codes
        #print(train_df["correct_model_encoded"].value_counts())

        category_mapping = dict(enumerate(train_df["correct_model"].astype('category').cat.categories))
        
        
        scaler = StandardScaler()
        mbpp_train_embeddings_scaled = scaler.fit_transform(mbpp_train_embeddings)

        lr = KNeighborsClassifier(n_neighbors=3)

        # Train on the full MBPP dataset
        lr.fit(mbpp_train_embeddings_scaled, train_df["correct_model_encoded"].values)
        
        
        accuracy, avg_energy = evaluate_router_humaneval(test_df, category_mapping, router_func=route_lr, router=lr)
        accuracies.append(accuracy)
        energies.append(avg_energy)
        n = len(test_df)
        print(f"Test set accuracy: {accuracy/n:.4f}, avg energy: {avg_energy/n:.2f}J")
        fold_accuracy = accuracy/n
        fold_energy = avg_energy/n
        fold_accuracies.append(fold_accuracy)
        fold_energies.append(fold_energy)
        
    n = len(df)/2
        
    average_accuracy = sum(accuracies) / n
    average_energy = sum(energies) / n


    print(average_accuracy, average_energy)
    return average_accuracy, average_energy
    

def fold_knn(e=0.5, k=5, random_state=42, fold_accuracies=None, fold_energies=None, neighbours=3):
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=random_state)
    accuracies = []
    energies = []

    for train_idx, test_idx in gkf.split(df, groups=groups):
        routers = []
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        emb_train = embeds[train_idx]


        # normalize using training split only
        energy_mean = X_train["avg_energy"].mean()
        energy_std  = X_train["avg_energy"].std()
        X_train["energy_norm"] = (X_train["avg_energy"] - energy_mean) / energy_std
        X_test["energy_norm"]  = (X_test["avg_energy"] - energy_mean) / energy_std

        ye_train = y_train - e * X_train["energy_norm"] 
        #ye_test = y_test - e * X_test["energy_norm"]
        
        X_train["model"].tolist()
        model_encoded = X_train["model"].astype('category').cat.codes.to_numpy()

        X_train_features = emb_train

        category_mapping = dict(enumerate(X_train["model"].astype('category').cat.categories))
        
        xgb_reg = KNeighborsRegressor(n_neighbors=neighbours)
        
        xgb_reg2 = KNeighborsRegressor(n_neighbors=neighbours)
        
        mask_model0 = model_encoded == 0
        mask_model1 = model_encoded == 1

        xgb_reg.fit(X_train_features[mask_model0], ye_train[mask_model0])
        xgb_reg2.fit(X_train_features[mask_model1], ye_train[mask_model1])
        
        routers.append(xgb_reg)
        routers.append(xgb_reg2)
        
        
        X_test_collapsed = X_test.drop_duplicates(subset=["task_id"]).reset_index(drop=True)
        accuracy, avg_energy = evaluate_router_humaneval(X_test_collapsed, category_mapping, router_func=route_xgb, router=routers)
        n = len(X_test_collapsed)
        print(f"Test set accuracy: {accuracy/n:.4f}, avg energy: {avg_energy/n:.2f}J")
        accuracies.append(accuracy)
        energies.append(avg_energy)
        
        fold_accuracy = accuracy/n
        fold_energy = avg_energy/n
        fold_accuracies.append(fold_accuracy)
        fold_energies.append(fold_energy)
        
    n = len(df)/2
        
    average_accuracy = sum(accuracies) / n
    average_energy = sum(energies) / n

    print(average_accuracy, average_energy)
    return average_accuracy, average_energy



def fold_xgb(e=0.5, k=5, random_state=42, fold_accuracies=None, fold_energies=None):
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=random_state)
    accuracies = []
    energies = []

    for train_idx, test_idx in gkf.split(df, groups=groups):
        routers = []
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        emb_train = embeds[train_idx]


        # normalize using training split only
        energy_mean = X_train["avg_energy"].mean()
        energy_std  = X_train["avg_energy"].std()
        X_train["energy_norm"] = (X_train["avg_energy"] - energy_mean) / energy_std
        X_test["energy_norm"]  = (X_test["avg_energy"] - energy_mean) / energy_std

        ye_train = y_train - e * X_train["energy_norm"] 
        
        X_train["model"].tolist()
        model_encoded = X_train["model"].astype('category').cat.codes.to_numpy()

        X_train_features = emb_train

        category_mapping = dict(enumerate(X_train["model"].astype('category').cat.categories))
        
        xgb_reg = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_reg2 = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        mask_model0 = model_encoded == 0
        mask_model1 = model_encoded == 1

        xgb_reg.fit(X_train_features[mask_model0], ye_train[mask_model0])
        xgb_reg2.fit(X_train_features[mask_model1], ye_train[mask_model1])
        
        routers.append(xgb_reg)
        routers.append(xgb_reg2)
        
        
        X_test_collapsed = X_test.drop_duplicates(subset=["task_id"]).reset_index(drop=True)
        accuracy, avg_energy = evaluate_router_humaneval(X_test_collapsed, category_mapping, router_func=route_xgb, router=routers)
        n = len(X_test_collapsed)
        print(f"Test set accuracy: {accuracy/n:.4f}, avg energy: {avg_energy/n:.2f}J")
        accuracies.append(accuracy)
        energies.append(avg_energy)
        
        fold_accuracy = accuracy/n
        fold_energy = avg_energy/n
        fold_accuracies.append(fold_accuracy)
        fold_energies.append(fold_energy)
        
    n = len(df)/2
        
    average_accuracy = sum(accuracies) / n
    average_energy = sum(energies) / n

    print(average_accuracy, average_energy)
    return average_accuracy, average_energy
    
def route_xgb(prompt, candidate_models, xgb):
    prompt_emb = embed_texts([prompt])

    predictions = []

    for code, model_name in candidate_models.items():
            
        pred = xgb[code].predict(prompt_emb)[0]
        predictions.append((model_name, pred))

    best_model_name, best_score = max(predictions, key=lambda x: x[1])
    return best_model_name, best_score


#global scaler
def route_lr(prompt, category_mapping, router):
    global scaler
    prompt_emb = embed_texts([prompt])
    prompt_emb_scaled = scaler.transform(prompt_emb)#normalize(prompt_emb) 
                
    lr = router
    c_pred = lr.predict(prompt_emb_scaled)[0]
            
    return category_mapping[c_pred], 123


def fold_lr(k=5, random_state=42, fold_accuracies=None, fold_energies=None):
    global scaler
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=random_state)
    accuracies = []
    energies = []

    for train_idx, test_idx in gkf.split(df, groups=groups):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]



        train_df = (
            df_train.groupby("task_id", group_keys=False)
            .apply(select_winner)
            .loc[:, ["task_id", "prompt", "model"]]
            .rename(columns={"model": "correct_model"})
            .reset_index(drop=True)
        )
        
        test_df = (
            df_test.groupby("task_id", group_keys=False)
            .apply(select_winner)
            .loc[:, ["task_id", "prompt", "model"]]
            .rename(columns={"model": "correct_model"})
            .reset_index(drop=True)
        )
        
        
        mbpp_train_prompts = train_df["prompt"].tolist()
        train_df["correct_model"].tolist()

        mbpp_train_embeddings = embed_texts(mbpp_train_prompts)
        train_df["correct_model_encoded"] = train_df["correct_model"].astype('category').cat.codes
        #print(train_df["correct_model_encoded"].value_counts())

        category_mapping = dict(enumerate(train_df["correct_model"].astype('category').cat.categories))
        
        
        scaler = StandardScaler()
        mbpp_train_embeddings_scaled = scaler.fit_transform(mbpp_train_embeddings)

        lr = LogisticRegression(
            max_iter=2000,
        )


        lr.fit(mbpp_train_embeddings_scaled, train_df["correct_model_encoded"].values)
        
        
        accuracy, avg_energy = evaluate_router_humaneval(test_df, category_mapping, router_func=route_lr, router=lr)
        accuracies.append(accuracy)
        energies.append(avg_energy)
        n = len(test_df)
        print(f"Test set accuracy: {accuracy/n:.4f}, avg energy: {avg_energy/n:.2f}J")
        fold_accuracy = accuracy/n
        fold_energy = avg_energy/n
        fold_accuracies.append(fold_accuracy)
        fold_energies.append(fold_energy)
        
    n = len(df)/2
        
    average_accuracy = sum(accuracies) / n
    average_energy = sum(energies) / n


    print(average_accuracy, average_energy)
    return average_accuracy, average_energy
    

def run_reproduce():
    print("Reproducing baseline results...")
    
    train_df = (
        df.groupby("task_id", group_keys=False)
        .apply(select_winner)
        .loc[:, ["task_id", "prompt", "model"]]
        .rename(columns={"model": "correct_model"})
        .reset_index(drop=True)
    )
    
    train_df["correct_model"].tolist()
    train_df["correct_model_encoded"] = train_df["correct_model"].astype('category').cat.codes

    category_mapping = dict(enumerate(train_df["correct_model"].astype('category').cat.categories))
    
    baseline_acc = []
    baseline_energy = []
    
    accuracy, avg_energy = evaluate_router_humaneval(train_df, category_mapping, router_func=None, router=None, mode="weak")
    baseline_acc.append(accuracy)
    baseline_energy.append(avg_energy)
    accuracy, avg_energy = evaluate_router_humaneval(train_df, category_mapping, router_func=None, router=None, mode="strong")
    baseline_acc.append(accuracy)
    baseline_energy.append(avg_energy)
    accuracy, avg_energy = evaluate_router_humaneval(train_df, category_mapping, router_func=None, router=None, mode="oracle")
    baseline_acc.append(accuracy)
    baseline_energy.append(avg_energy)
    
    np.save("Data/baseline_acc.npy", np.array(baseline_acc)/len(train_df))
    np.save("Data/baseline_energy.npy", np.array(baseline_energy)/len(train_df))
    
    
    print("Reproducing router results...")    
    accuracies_knnc = []
    energies_knnc = []
    accuracies_knn = []
    energies_knn = []
    accuracies_lr = []
    energies_lr = []
    accuracies_xgb = []
    energies_xgb = []
    xgb_lambda_accuracies = []
    xgb_lambda_energies = []
    knn_lambda_accuracies = []
    knn_lambda_energies = []
    repetitions = 10
    
    for i in range(repetitions):
        print("Repetition:", i+1)
        fold_accuracies = []
        fold_energies = []
        accuracy, energy = fold_knnc(k=5, random_state=42+i, fold_accuracies=fold_accuracies, fold_energies=fold_energies)
        accuracies_knnc.append(accuracy)
        energies_knnc.append(energy)
        accuracy, energy = fold_knn(e=0.5, k=5, random_state=42+i, fold_accuracies=fold_accuracies, fold_energies=fold_energies, neighbours=3)
        accuracies_knn.append(accuracy)
        energies_knn.append(energy)
        accuracy, energy = fold_xgb(e=0.5, k=5, random_state=42+i, fold_accuracies=fold_accuracies, fold_energies=fold_energies)
        accuracies_xgb.append(accuracy)
        energies_xgb.append(energy)
        accuracy, energy = fold_lr(k=5, random_state=42+i, fold_accuracies=fold_accuracies, fold_energies=fold_energies)
        accuracies_lr.append(accuracy)
        energies_lr.append(energy)
    

    
    np.save("Data/accuracies_lr.npy", np.array(accuracies_lr))
    np.save("Data/energies_lr.npy", np.array(energies_lr))
    np.save("Data/accuracies_xgb.npy", np.array(accuracies_xgb))
    np.save("Data/energies_xgb.npy", np.array(energies_xgb))
    np.save("Data/accuracies_knn.npy", np.array(accuracies_knn))
    np.save("Data/energies_knn.npy", np.array(energies_knn))
    np.save("Data/accuracies_knnc.npy", np.array(accuracies_knnc))
    np.save("Data/energies_knnc.npy", np.array(energies_knnc))
    print("Finished reproducing main results. Starting energy factor evaluation...")
    #Lambdas
    
    xgb_lambda_accuracies = []
    xgb_lambda_energies = []
    knn_lambda_accuracies = []
    knn_lambda_energies = []

    for ee in range (0, 11, 1):
        print("Energy factor lambda:", ee/10)
        e = ee/10

        accuracies_xgb = []
        energies_xgb = []
        accuracies_knn = []
        energies_knn = []
        fold_accuracies = []
        fold_energies= []

        for i in range(repetitions):
            accuracy, energy = fold_xgb(e=e, k=5, random_state=42+i, fold_accuracies=fold_accuracies, fold_energies=fold_energies)
            accuracies_xgb.append(accuracy)
            energies_xgb.append(energy)
            accuracy, energy = fold_knn(e=e, k=5, random_state=42+i, fold_accuracies=fold_accuracies, fold_energies=fold_energies, neighbours=3)
            accuracies_knn.append(accuracy)
            energies_knn.append(energy)
        
        
        xgb_acc_avg = np.mean(accuracies_xgb)
        xgb_energy_avg = np.mean(energies_xgb)
        
        xgb_lambda_accuracies.append(xgb_acc_avg)
        xgb_lambda_energies.append(xgb_energy_avg)
        
        knn_acc_avg = np.mean(accuracies_knn)
        knn_energy_avg = np.mean(energies_knn)
        
        knn_lambda_accuracies.append(knn_acc_avg)
        knn_lambda_energies.append(knn_energy_avg)
    
    np.save("Data/xgb_lambda_accuracies.npy", np.array(xgb_lambda_accuracies))
    np.save("Data/xgb_lambda_energies.npy", np.array(xgb_lambda_energies))
    np.save("Data/knn_lambda_accuracies.npy", np.array(knn_lambda_accuracies))
    np.save("Data/knn_lambda_energies.npy", np.array(knn_lambda_energies))
    print("Finished reproducing data. Data stored in Data/ folder.")
    
    run_plot()
    
def run_plot():
    print("Calculating results and generating plots...")
    accuracies_lr = np.load("Data/accuracies_lr.npy").tolist()
    energies_lr = np.load("Data/energies_lr.npy").tolist()
    accuracies_xgb = np.load("Data/accuracies_xgb.npy").tolist()
    energies_xgb = np.load("Data/energies_xgb.npy").tolist()
    accuracies_knn = np.load("Data/accuracies_knn.npy").tolist()
    energies_knn = np.load("Data/energies_knn.npy").tolist()
    accuracies_knnc = np.load("Data/accuracies_knnc.npy").tolist()
    energies_knnc = np.load("Data/energies_knnc.npy").tolist()
    
    baseline_acc = np.load("Data/baseline_acc.npy").tolist()
    baseline_energy = np.load("Data/baseline_energy.npy").tolist()
    
    # print("DEBUG")
    # print(accuracies_knn, energies_knn)
    
    
    
    
    # Data
    methods = [
        "XGBoost",
        "Logistic Regression",
        "DeepSeek 1.3B",
        "Qwen 3B",
        "Oracle",
        "KNN", #KNN Regression
        "KNN" #KNN Classifier
    ]

    accuracy = [
        np.mean(accuracies_xgb),
        np.mean(accuracies_lr),
        baseline_acc[0],
        baseline_acc[1],
        baseline_acc[2],
        np.mean(accuracies_knn),
        np.mean(accuracies_knnc)
    ]

    energy = [
        np.mean(energies_xgb),
        np.mean(energies_lr),
        baseline_energy[0],
        baseline_energy[1],
        baseline_energy[2],
        np.mean(energies_knn),
        np.mean(energies_knnc)
    ]


    new_methods = [methods[1], methods[6], "KNN Classifier", "KNN Regressor"]
    new_acc = [accuracy[1], accuracy[6], accuracy[0], accuracy[5]]
    new_energy = [energy[1], energy[6], energy[0], energy[5]]

    interpolated = linear_interpolate(new_acc, baseline_acc[0], baseline_energy[0], baseline_acc[1], baseline_energy[1])

    a = new_energy
    b = interpolated
    reductions = percent_reduction(b, a)
    # print("accs")
    # print(new_acc)
    # print("energies")
    # print(new_energy)
    # print("reductions")
    # print(percent_reduction(b, a))
    # print(interpolated)

    # print(b)
    
    for i in range(len(new_methods)):
        print(f"{new_methods[i]}: Accuracy: {new_acc[i]:.4f}, Energy: {new_energy[i]:.2f}J, Reduction vs Baseline: {reductions[i]:.2f}%")

    plt.figure(figsize=(8,6))

    # ---- Plot individual points with custom colors ----
    colors = [
        "orange",   # XGB Router
        "green",   # LR Router
        "blue",     # All Low
        "blue",    # All High
        "red",       # Oracle
        "orange",
        "green"
    ]

    for i in range(len(methods)):
        plt.scatter(accuracy[i], energy[i], s=50, color=colors[i])
        plt.annotate(methods[i],
                    (accuracy[i], energy[i]),
                    xytext=(6,6),
                    textcoords="offset points")

    # ---- Interpolation line between All Low and All High ----
    plt.plot(
        [accuracy[2], accuracy[3]],
        [energy[2], energy[3]],
        linestyle="--",
        color="blue",
        alpha=0.7
    )

    # ---- Custom legend ----
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Regression Routers',
            markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Classifier Routers',
            markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Base Models',
            markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Oracle',
            markerfacecolor='red', markersize=8),
    ]

    plt.legend(handles=legend_elements)

    plt.xlim(0.5, 0.75)

    plt.xlabel("Accuracy")
    plt.ylabel("Energy (Joules)")
    plt.title("Router performance vs Baseline")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./images/RouterAcc.pdf", format="pdf")
#    plt.show()
    
    
    
    # ---- XGB e parameter values ----
    e_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]



    xgb_accuracy = np.load("Data/xgb_lambda_accuracies.npy").tolist()
    xgb_energy = np.load("Data/xgb_lambda_energies.npy").tolist()
    knn_accuracy = np.load("Data/knn_lambda_accuracies.npy").tolist()
    knn_energy = np.load("Data/knn_lambda_energies.npy").tolist()

    print("XGB Accuracies:", xgb_accuracy)
    print("XGB Energies:", xgb_energy)
    print("KNN Accuracies:", knn_accuracy)
    print("KNN Energies:", knn_energy)
    # ---- Baseline line ----


    plt.figure(figsize=(8,6))

    # ---- Plot XGB points ----
    plt.plot(xgb_accuracy, xgb_energy, marker='o', color='orange', label='XGB Regression Router $\\lambda$')
    plt.plot(knn_accuracy, knn_energy, marker='o', color='green', label='KNN Regression Router $\\lambda$')

    # ---- Annotate selected λ values directly below the points ----
    highlight_lambdas = [0, 0.1, 0.3, 0.5, 1]

    for acc, en, e in zip(xgb_accuracy, xgb_energy, e_values):
        if e in highlight_lambdas:
            if e == 0 or e == 1:
                label = f"$\\lambda = {e}$"
            else:
                label = f"{e}"
            plt.annotate(label,
                        (acc, en),
                        xytext=(0, -12),
                        textcoords='offset points',
                        fontsize=9,
                        ha='center')
            


    # ---- Plot baseline line ----
    plt.plot(baseline_acc[0:2], baseline_energy[0:2], linestyle='--', color='blue', alpha=0.7, label='Baseline')

    # ---- Plot baseline endpoints as points ----
    plt.scatter(baseline_acc[0:2], baseline_energy[0:2], color='blue', s=50, zorder=5)

    # ---- Annotate baseline points ----
    plt.annotate("Deepseek",
                (baseline_acc[0], baseline_energy[0]),
                xytext=(0, -12),
                textcoords='offset points',
                fontsize=9,
                ha='center')

    plt.annotate("Qwen",
                (baseline_acc[1], baseline_energy[1]),
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=9,
                ha='center')

    plt.xlabel("Accuracy")
    plt.ylabel("Energy (Joules)")
    plt.title("Regression Routers $\\lambda$ Parameter Trade-off vs Baseline")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./images/RouterLambda.pdf", format="pdf")
#    plt.show()








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduction pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reproduce", action="store_true", help="Run full reproduction")
    group.add_argument("--plot", action="store_true", help="Only generate plots")

    args = parser.parse_args()

    if args.reproduce:
        run_reproduce()
    elif args.plot:
        run_plot()
