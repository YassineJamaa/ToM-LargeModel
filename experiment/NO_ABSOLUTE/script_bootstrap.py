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
import json
from functools import partial

from benchmark import FanToMPromptEngineering, ToMiPromptEngineering, MMToMQAPromptEngineering, MATHBenchmark, OpenToMPromptEngineering
from src import (
                AssessmentTopK,
                Assessmentv6,
                setup_environment,
                 ImportModel,
                 load_chat_template,
                 LayerUnits,
                 LocImportantUnits,
                 LocNoAbs,
                 ZeroingAblation,
                 ToMLocDataset,
                 AverageTaskStimuli,
                 MeanImputation,
                 ExtendedTomLocGPT4,
                 MDLocDataset,
                 get_masked_kbot,
                 get_masked_ktop)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--localizer", type=str, default="ToM", help="Either: MD or ToM")
    parser.add_argument("--n_bootstrap", type=int, default=100, help="Bootstraping Dataset")
    parser.add_argument("--saving_interval", type=int, default=5, help="Saving Interval")
    parser.add_argument("--benchmark_name", type=str, default="ToMi", choices=["variant_FanToM", "FanToM", "MATH", "OpenToM", "ToMi"], help="Either: MD or ToM")
    parser.add_argument("--percentage", type=float, default=0)
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
                answer_letter_list = existing_data.get("answer_letter_list", [])
                predictions = existing_data.get("predictions", [])
                seed_list = existing_data.get("seed", [])
                seed = seed_list[-1]+1
            except json.JSONDecodeError:
                print("Warning: JSON file is empty or corrupted. Starting fresh.")

    else:
        answer_letter_list = []
        predictions = []
        seed_list = []
        seed=0
    return answer_letter_list, predictions, seed, seed_list

def saving_result(seed_list, answer_letter_list, predictions, json_file):
    # Prepare dictionary to store in JSON
    result_data = {
        "seed": seed_list,
        "answer_letter_list": answer_letter_list,
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
    n_bootstrap = args.n_bootstrap
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
    loc = LocNoAbs("classic", units.data_activation)
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=llm,
                        ablation=zeroing,
                        device=device)
    ##############################################################

    ##### Set Json file & Benchmark & Mask ##############################
    subset = None
    if benchmark_name=="variant_FanToM":
        bootstrap_canvas = benchmark_config["FanToM"](subset=subset, prompt_type="variant") # RESET TO NONE AFTER
    else:
        bootstrap_canvas = benchmark_config[benchmark_name](subset=subset) # RESET TO NONE AFTER
    #######################################################################
    data = bootstrap_canvas.data.copy()
    json_file = os.path.join("Result", "Final-Analysis", "Bootstrap", model_name, f"{benchmark_name}_{percentage:.2%}_lesion.json")
    mask = get_masked_ktop(loc, percentage)
    ##############################################################

    ############ Initiliaze les données ############################
    answer_letter_list, predictions, seed, seed_list = initialize_data(json_file)
    ################################################################


    ################################################################
    ##################### MAIN SCRIPT ##############################
    ################################################################
    print(f"{model_name.upper()}:")
    print(f"Benchmark: {type(bootstrap_canvas).__name__} analysis...")
    print(f"Seed range from {seed} to {n_bootstrap}...")
    for i in range(seed, n_bootstrap):
        # Set the bootstrap
        boostrap_data = data.sample(n=len(data), replace=True, random_state=i).reset_index(drop=True)
        bootstrap_canvas.data = boostrap_data
        answer_letter_list.append(bootstrap_canvas.data.answer_letter.tolist())
        seed_list.append(i)

        # Assessement
        res = assess.experiment(bootstrap_canvas, mask, num_random=0, no_lesion=False)
        acc = (res["predict_ablate_top"].str.lower() == res["answer_letter"]).mean()
        predictions.append(res["predict_ablate_top"].tolist())
        print(f"Accuracy = {acc:.2%}\n")

        if i%saving_interval==0 and i>0:
            print(f"Saving results at iteration {i}...")
            saving_result(seed_list=seed_list,
                        answer_letter_list=answer_letter_list,
                        predictions=predictions,
                        json_file=json_file)

    saving_result(seed_list=seed_list,
                answer_letter_list=answer_letter_list,
                predictions=predictions,
                json_file=json_file)
    print("Process completed.")

if __name__ == "__main__":
    main()



