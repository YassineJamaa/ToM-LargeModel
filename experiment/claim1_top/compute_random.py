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

from benchmark import FanToMPromptEngineering, ToMiPromptEngineering, MMToMQAPromptEngineering, MATHBenchmark, OpenToMPromptEngineering
from src import (
                AssessmentTopK,
                Assessmentv6,
                setup_environment,
                 ImportModel,
                 load_chat_template,
                 LayerUnits,
                 LocImportantUnits,
                 ZeroingAblation,
                 ToMLocDataset,
                 AverageTaskStimuli,
                 MeanImputation,
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
    parser.add_argument("--localizer", type=str, default="ToM", help="Either: MD or ToM")
    parser.add_argument("--n_random", type=int, default=15, help="Bootstraping Dataset")
    parser.add_argument("--saving_interval", type=int, default=5, help="Saving Interval")
    parser.add_argument("--benchmark_name", type=str, default="ToMi", choices=["FanToM", "MATH", "OpenToM", "ToMi"], help="Either: MD or ToM")
    parser.add_argument("--percentage", type=float, default=0.01)
    return parser.parse_args()

benchmark_config = {
    "ToMi": ToMiPromptEngineering,
    "MATH": MATHBenchmark,
    "FanToM": partial(FanToMPromptEngineering, is_full=False),
    "OpenToM": OpenToMPromptEngineering
}

def initialize_data(json_file):
    # Check if the JSON file already exists
    if os.path.exists(json_file):
        print(f"Existing JSON file found: {json_file}. Loading previous results...")
        with open(json_file, "r") as f:
            try:
                existing_data = json.load(f)
                predictions = existing_data.get("random_predictions", [])
                seed_list = existing_data.get("seed", [])
                seed = seed_list[-1]+1
            except json.JSONDecodeError:
                print("Warning: JSON file is empty or corrupted. Starting fresh.")

    else:
        predictions = []
        seed_list = []
        seed=0
    return predictions, seed, seed_list

def saving_result(seed_list, answer_letter, predictions, json_file):
    # Prepare dictionary to store in JSON
    result_data = {
        "seed": seed_list,
        "answer_letter": answer_letter,
        "predictions": predictions
    }
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    # Write to JSON file
    with open(json_file, "w") as f:
        json.dump(result_data, f, indent=4)

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
    saving_interval = args.saving_interval
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

    ######### Set the FRAMEWORK ####################################
    if localizer == "MD":
        loc_data = MDLocDataset()
    elif localizer == "ToM":
        loc_data = ToMLocDataset()

    units = LayerUnits(import_model = llm, localizer = loc_data, device = device)
    loc = LocImportantUnits("classic", units.data_activation)
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=llm,
                        ablation=zeroing,
                        device=device)
    ##############################################################
    
    ########## Initialize #########################################
    benchmark = benchmark_config[benchmark_name](subset=None) # RESET TO NONE AFTER
    answer_letter = benchmark.data.answer_letter.tolist()
    json_file = os.path.join("Result", "CLAIM1", "RANDOM_MASK", model_name, f"{type(benchmark).__name__}_{percentage:.2%}.json")
    predictions, seed, seed_list = initialize_data(json_file)
    ###############################################################

    ################################################################
    ##################### MAIN SCRIPT ##############################
    ################################################################
    print(f"{model_name.upper()}:")
    print(f"Benchmark: {type(benchmark).__name__} analysis...")
    print(f"Seed range from {seed} to {n_random}...")
    random_predictions = []
    for i in range(seed, n_random):
        mask_random = get_masked_random(loc, percentage, seed=i)
        res_random = assess.experiment(benchmark, mask_random, num_random=0, no_lesion=False)
        random_predictions.append(res_random["predict_ablate_top"].tolist())
        accs = (res_random["predict_ablate_top"].str.lower()==res_random["answer_letter"]).mean()
        print(f"Seed={i}, Random-{percentage:.2%} Accuracy={accs:.2%}")
        seed_list.append(i)

        if i%saving_interval==0 and i>0:
            print(f"Saving results at iteration {i}...")
            saving_result(seed_list=seed_list,
                        answer_letter=answer_letter,
                        predictions=random_predictions,
                        json_file=json_file)
    
    saving_result(seed_list=seed_list,
                answer_letter=answer_letter,
                predictions=random_predictions,
                json_file=json_file)
    print("Process completed.")

if __name__ == "__main__":
    main()



