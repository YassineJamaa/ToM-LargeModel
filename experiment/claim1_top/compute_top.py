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
                 LocImportantUnits,
                 ZeroingAblation,
                 ToMLocDataset,
                 AverageTaskStimuli,
                 MeanImputation,
                 LangLocDataset,
                 ExtendedTomLocGPT4,
                 MDLocDataset,
                 get_masked_ktop)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--localizer", type=str, default="ToM", help="Either: MD or ToM or language")
    parser.add_argument("--benchmark_name", type=str, default="ToMi", choices=["FanToM", "MATH", "OpenToM", "ToMi"], help="Either: MD or ToM")
    parser.add_argument("--percentage", type=float, default=0)
    return parser.parse_args()

benchmark_config = {
    "ToMi": ToMiPromptEngineering,
    "MATH": MATHBenchmark,
    "FanToM": partial(FanToMPromptEngineering, is_full=False),
    "OpenToM": OpenToMPromptEngineering
}

def saving_result(benchmark, predictions, json_file):
    # Prepare dictionary to store in JSON
    result_data = {
        "benchmark": benchmark.data.to_dict(orient="records"),
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
    elif localizer == "Language":
        loc_data = LangLocDataset()

    units = LayerUnits(import_model = llm, localizer = loc_data, device = device)
    loc = LocImportantUnits("classic", units.data_activation)
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=llm,
                        ablation=zeroing,
                        device=device)
    ##############################################################

    ########## Initialize #########################################
    benchmark = benchmark_config[benchmark_name](subset=None) # RESET TO NONE AFTER
    json_file = os.path.join("Result", "CLAIM1", f"TOP_MASK_{localizer}", model_name, f"{type(benchmark).__name__}_{percentage:.2%}.json")
    ###############################################################

    ################################################################
    ##################### MAIN SCRIPT ##############################
    ################################################################
    print(f"{model_name.upper()}:")
    print(f"Benchmark: {type(benchmark).__name__} analysis...")
    mask_bot = get_masked_ktop(loc, percentage)
    res_bot = assess.experiment(benchmark, mask_bot, num_random=0, no_lesion=False)
    accs = (res_bot["predict_ablate_top"].str.lower()==res_bot["answer_letter"]).mean()
    print(f"Top-{percentage:.2%} Accuracy={accs:.2%}")

    saving_result(benchmark, res_bot["predict_ablate_top"].tolist(), json_file)

    print("Process completed.")

if __name__ == "__main__":
    main()