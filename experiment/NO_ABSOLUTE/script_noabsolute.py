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
    parser.add_argument("--benchmark_name", type=str, default="variant_FanToM", choices=["variant_FanToM", "FanToM", "MATH", "OpenToM", "ToMi"], help="Either: MD or ToM")
    return parser.parse_args()

benchmark_config = {
    "ToMi": ToMiPromptEngineering,
    "MATH": MATHBenchmark,
    "FanToM": partial(FanToMPromptEngineering, is_full=False),
    "OpenToM": OpenToMPromptEngineering
}

def saving_result(benchmark, predictions, percentages, json_file):
    result_data = {
        "benchmark": benchmark.data.to_dict(orient="records"),
        "percentages": percentages.tolist(),
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
    if benchmark_name=="variant_FanToM":
        benchmark = benchmark_config["FanToM"](subset=None, prompt_type="variant") # RESET TO NONE AFTER
    else:
        benchmark = benchmark_config[benchmark_name](subset=None) # RESET TO NONE AFTER

    json_file = os.path.join("Result", "Final-Analysis", "No-absolute", f"BOT_MASK_{localizer}", model_name, f"{benchmark_name}_bottom_lesion.json")
    percentages = np.array([0.005, 0.01, 0.02])
    ###############################################################

    ################################################################
    ##################### MAIN SCRIPT ##############################
    ################################################################
    print(f"{model_name.upper()}:")
    print(f"Benchmark: {type(benchmark).__name__} analysis...")
    predictions = []
    for percentage in percentages:
        mask = get_masked_kbot(noabs_loc, percentage)
        res = assess.experiment(benchmark, mask, num_random=0, no_lesion=False)

        outliers = [prediction for prediction in res["predict_ablate_top"].tolist() if prediction.lower() not in ["a", "b"]]
        print(f"Nb. of Outliers {len(outliers)}")
        print(f" Accuracy Diff-{percentage:.2%} Lesion: {(res["predict_ablate_top"].str.lower() == res["answer_letter"]).mean():.2%}")

        predictions.append(res["predict_ablate_top"].tolist())
    
    saving_result(benchmark, predictions, percentages, json_file)
    print("Process completed.")

if __name__ == "__main__":
    main()







