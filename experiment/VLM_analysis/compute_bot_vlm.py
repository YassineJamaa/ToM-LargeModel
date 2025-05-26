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

from benchmark import MMStarBenchmark, MathVistaBenchmark
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
                 get_masked_kbot,
                 get_masked_ktop,
                 get_masked_random)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForImageTextToText", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--localizer", type=str, default="ToM", help="Either: MD or ToM or language")
    parser.add_argument("--benchmark_name", type=str, default="MMStar", choices=["MMStar", "MathVista"], help="Either: MD or ToM")
    parser.add_argument("--percentage", type=float, default=0)
    return parser.parse_args()

benchmark_config = {
    "MMStar": MMStarBenchmark,
    "MathVista": MathVistaBenchmark,
}

def saving_result(benchmark, predictions, json_file):
    # Prepare dictionary to store in JSON
    columns_to_drop = ['image', 'raw_byte_img', "decoded_image"]
    dataframe = benchmark.data.copy()
    dataframe = dataframe.drop(columns=[col for col in columns_to_drop if col in dataframe.columns])
    result_data = {
        "benchmark": dataframe.to_dict(orient="records"),
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
    model_type = "VLM"
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
    vlm = set_model(**model_args)

    ######### Set the FRAMEWORK ####################################
    if localizer == "MD":
        loc_data = MDLocDataset()
    elif localizer == "ToM":
        loc_data = ToMLocDataset()
    elif localizer == "Language":
        loc_data = LangLocDataset()

    units = LayerUnits(import_model = vlm, localizer = loc_data, device = device)
    loc = LocImportantUnits("classic", units.data_activation)
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=vlm,
                        ablation=zeroing,
                        device=device)
    ##############################################################

    ########## Initialize #########################################
    benchmark = benchmark_config[benchmark_name](subset=None) # RESET TO NONE AFTER
    json_file = os.path.join("Result_VLM", "CLAIM2", f"BOT_MASK_{localizer}", model_name, f"{type(benchmark).__name__}_{percentage:.2%}.json")
    ###############################################################

    ################################################################
    ##################### MAIN SCRIPT ##############################
    ################################################################
    print(f"{model_name.upper()}:")
    print(f"Benchmark: {type(benchmark).__name__} analysis...")
    mask_bot = get_masked_kbot(loc, percentage)
    res_bot = assess.experiment(benchmark, mask_bot, num_random=0, no_lesion=False)
    accs = (res_bot["predict_ablate_top"].str.lower()==res_bot["answer_letter"]).mean()
    print(f"Bot-{percentage:.2%} Accuracy={accs:.2%}")

    saving_result(benchmark, res_bot["predict_ablate_top"].tolist(), json_file)
    ##################################################################

    print("Process completed.")

if __name__ == "__main__":
    main()