import numpy as np
import pandas as pd
import json

def compute_diff(df, filter_variable):
    unique_values = df[filter_variable].unique()
    diff = {f"{value}": {"top_k": []} for value in unique_values}

    for value in unique_values:
        category_df = df[df[filter_variable] == value]
        top_k_diff = (category_df["answer"] == category_df["predict_ablate_top_1"]).mean() - (
            category_df["answer"] == category_df["predict_no_ablation"]).mean()
        diff[f"{value}"]["top_k"].append(top_k_diff)

    random_cols = [col for col in df.columns if col.startswith("predict_ablate_random")]
    if random_cols:
        for value in unique_values:
            diff[f"{value}"]["random"] = []
            category_df = df[df[filter_variable] == value]
            for random_col in random_cols:
                random_diff = (category_df["answer"] == category_df[random_col]).mean() - (
                    category_df["answer"] == category_df["predict_no_ablation"]).mean()
                diff[f"{value}"]["random"].append(random_diff)

    return diff

def compute_mean_and_std(values):
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

def save_diff(df, filter_variable, file_path="diff.json"):
    if filter_variable is None:
        raise ValueError("The filter_variable must be specified.")

    diff = compute_diff(df, filter_variable)

    with open(file_path, "w") as json_file:
        json.dump(diff, json_file, indent=4)
    print(f"Diff saved to {file_path}")
