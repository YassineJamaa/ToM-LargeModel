""" This script compute the performance of random units for the percentages investigate for "No absolute" values.
    The random mask is done as usual.

 """

import os
from dotenv import load_dotenv
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import torch
from functools import partial
import json
import pickle

from benchmark import FanToMPromptEngineering, ToMiPromptEngineering, MMToMQAPromptEngineering, MATHBenchmark, OpenToMPromptEngineering
from src import (
                AssessmentTopK,
                Assessmentv6,
                setup_environment,
                 ImportModel,
                 load_chat_template,
                 LayerUnits,
                 LocNoAbs,
                 LocImportantUnits,
                 ZeroingAblation,
                 ToMLocDataset,
                 AverageTaskStimuli,
                 MeanImputation,
                 LangLocDataset,
                 ExtendedTomLocGPT4,
                 MDLocDataset,
                 get_masked_kbot,
                 get_masked_ktop,
                 get_masked_random)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--localizer", type=str, default="ToM", help="Either: MD or ToM or language")
    parser.add_argument("--benchmark_name", type=str, default="ToMi", choices=["variant_FanToM", "FanToM", "MATH", "OpenToM", "ToMi"], help="Either: MD or ToM")
    parser.add_argument("--n_random", type=int, default=15, help="Bootstraping Dataset")
    return parser.parse_args()

benchmark_config = {
    "ToMi": ToMiPromptEngineering,
    "MATH": MATHBenchmark,
    "FanToM": partial(FanToMPromptEngineering, is_full=False),
    "OpenToM": OpenToMPromptEngineering
}

def saving_result(seed_list, answer_letter, predictions, percentages, json_file):
    # Prepare dictionary to store in JSON
    result_data = {
        "seed": seed_list,
        "answer_letter": answer_letter,
        "predictions": predictions,
        "percentages": percentages.tolist()
    }
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    # Write to JSON file
    with open(json_file, "w") as f:
        json.dump(result_data, f, indent=4)

def initialize_data(json_file):
    # Check if the JSON file already exists
    if os.path.exists(json_file):
        print(f"Existing JSON file found: {json_file}. Loading previous results...")
        with open(json_file, "r") as f:
            try:
                existing_data = json.load(f)
                predictions = existing_data.get("random_predictions", [])
                seed_list = existing_data.get("seed", [])
                percentages = existing_data.get("percentages", [])
                percentage_idx = len(percentages)
            except json.JSONDecodeError:
                print("Warning: JSON file is empty or corrupted. Starting fresh.")

    else:
        predictions = []
        seed_list = []
        percentage_idx = 0
    return predictions, percentage_idx, seed_list

def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_name = model_checkpoint.split("/")[-1]
    model_func = args.model_func
    model_type = "LLM"
    cache = args.cache
    localizer = args.localizer
    benchmark_name = args.benchmark_name
    n_random = args.n_random

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

    if localizer == "MD":
        loc_data = MDLocDataset()
    elif localizer == "ToM":
        loc_data = ToMLocDataset()
    elif localizer == "Language":
        loc_data = LangLocDataset()
    
    units = LayerUnits(import_model = llm, localizer = loc_data, device = device)
    noabs_loc = LocNoAbs("classic", units.data_activation)
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=llm,
                        ablation=zeroing,
                        device=device)
    
    ########## Initialize #########################################
    subset = None
    if benchmark_name=="variant_FanToM":
        benchmark = benchmark_config["FanToM"](subset=subset, prompt_type="variant") # RESET TO NONE AFTER
    else:
        benchmark = benchmark_config[benchmark_name](subset=subset) # RESET TO NONE AFTER

    answer_letter = benchmark.data.answer_letter.tolist()
    json_file = os.path.join("Result", "Appendix-Result", "Random", model_name, f"{benchmark_name}_random_lesion.json")
    predictions, percentage_idx, seed_list = initialize_data(json_file)
    # percentages = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
    percentages = np.array([0.005])
    ###############################################################

    ################################################################
    ##################### MAIN SCRIPT ##############################
    ################################################################
    print(f"{model_name.upper()}:")
    print(f"Benchmark: {type(benchmark).__name__} analysis...")
    predictions = []
    for percentage in percentages[percentage_idx:]:
        seed_tmp = []
        predictions_tmp = []
        print(f"Random-{percentage:.2%}...")
        for i in range(n_random):
            mask_random = get_masked_random(noabs_loc, percentage, seed=i)
            res = assess.experiment(benchmark, mask_random, num_random=0, no_lesion=False)
            outliers = [prediction for prediction in res["predict_ablate_top"].tolist() if prediction.lower() not in ["a", "b"]]
            print(f"Nb. of Outliers {len(outliers)}")
            print(f"Random-{percentage:.2%} Lesion: {(res["predict_ablate_top"].str.lower() == res["answer_letter"]).mean():.2%}")
            predictions_tmp.append(res["predict_ablate_top"].tolist())
            seed_tmp.append(i)
        predictions.append(predictions_tmp)
        seed_list.append(seed_tmp)
        saving_result(
            seed_list=seed_list,
            answer_letter=answer_letter,
            predictions=predictions,
            percentages=percentages,
            json_file=json_file
        )
    print("Process completed.")

if __name__ == "__main__":
    main()
