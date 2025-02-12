import os
from dotenv import load_dotenv
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import torch
import json

from benchmark import BenchmarkMMToMQA, BenchmarkToMi, BenchmarkOpenToM, BenchmarkFanToM, BenchmarkBaseline, BenchmarkVisionText, ToMiPromptEngineering, MMToMQAPromptEngineering
from src import (
                Assessmentv4,
                setup_environment,
                 ImportModel,
                 load_chat_template,
                 LayerUnits,
                 LocImportantUnits,
                 ZeroingAblation,
                 ToMLocDataset,
                 AverageTaskStimuli,
                 MeanImputation,
                 ExtendedTomLocGPT4,)
from analysis import set_model


def get_masked_kbot(classic_loc, percentage):
    if percentage==0:
        return np.zeros_like(classic_loc.t_values, dtype=np.int64)
    num_top_elements = int(classic_loc.t_values.size * percentage)

    # Flatten the matrix, find the threshold value for the top 1%
    flattened_matrix = classic_loc.t_values.flatten()

    # Replace NaN with a sentinel value (e.g., -inf)
    flattened_matrix = np.nan_to_num(flattened_matrix, nan=-np.inf)
    
    # Filter out -inf values
    filtered_matrix = flattened_matrix[flattened_matrix != -np.inf]
    if len(filtered_matrix) > num_top_elements:
        threshold_value = np.partition(filtered_matrix, num_top_elements)[num_top_elements]
    else:
        threshold_value = -np.inf  # or handle this case as needed      
    # Create a binary mask where 1 represents the top 1% elements, and 0 otherwise
    mask_units = np.where(classic_loc.t_values <= threshold_value, 1, 0)
    return mask_units

def create_submask(mask: np.ndarray, sub_pcts: float, seed: int):
    """
    Creates a submask by keeping a fixed percentage of activated (`1`) units.
    
    :param mask: Binary numpy array (0s and 1s)
    :param sub_pcts: Percentage of activated units to keep (relative to total size)
    :param seed: Random seed for reproducibility
    :return: A new binary submask with the requested proportion of `1`s
    """
    rng = np.random.default_rng(seed)
    total_units = mask.size
    keep_activate_units = int(total_units * sub_pcts)
    activate_indices = np.argwhere(mask==1)
    selected_indices = activate_indices[rng.choice(len(activate_indices), keep_activate_units, replace=False)]
    submask = np.zeros_like(mask)
    submask[tuple(selected_indices.T)] = 1
    return submask

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--percentage", type=float, default=0.004, help="")
    return parser.parse_args()

def main():
    # Parse command-line
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_func = args.model_func
    model_type = "LLM"
    cache = args.cache
    percentage = args.percentage

    token, cache_dir = setup_environment(cache=cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = {
        "model_type": model_type,
        "model_checkpoint": model_checkpoint,
        "token_hf": token,
        "cache_dir": cache_dir,
        "model_func": model_func,
    }
    llm = set_model(**model_args)

    classic_tom = ToMLocDataset()
    classic_units = LayerUnits(import_model = llm, localizer = classic_tom, device = device)
    classic_loc = LocImportantUnits("classic", classic_units.data_activation)
    benchmark = ToMiPromptEngineering()
    zeroing = ZeroingAblation()
    assess = Assessmentv4(import_model=llm,
                        ablation=zeroing,
                        device=device)
    
    # percentage = 0.01
    diff_units = [0.0001, 0.001, 0.0015]
    seed = 42
    mask_bot = get_masked_kbot(classic_loc, percentage)
    dict_data = {f"{percentage:.2%}": {}}
    for diff_unit in diff_units:
        sub_pct=percentage-diff_unit
        bottom_accs = []
        for i in range(75):
            submask = create_submask(mask_bot, sub_pct, seed+i)
            res = assess.experiment(benchmark, submask, num_random=0, no_lesion=False)
            bot_acc = (res["predict_ablate_top"] == res["answer_letter"]).mean()
            bottom_accs.append(bot_acc)

        formatted_percentage = f"{diff_unit:.2%}"
        cleaned_percentage = formatted_percentage.rstrip('0').rstrip('.')
        dict_data[f"{percentage:.2%}"][f"diff_units_{cleaned_percentage}"]=bottom_accs
    model_name = model_checkpoint.split("/")[-1]
    benchmark_name = type(benchmark).__name__
    json_filename = os.path.join("Result", "Sub_bottom", model_name, f"{benchmark_name}_{percentage:.2%}.json")
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    with open(json_filename, "w") as json_file:
        json.dump(dict_data, json_file, indent=4)

if __name__ == "__main__":
    main()



