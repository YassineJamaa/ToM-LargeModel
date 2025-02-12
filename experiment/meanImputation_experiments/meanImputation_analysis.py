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

from benchmark import BenchmarkMMToMQA, BenchmarkToMi, BenchmarkOpenToM, BenchmarkFanToM, BenchmarkBaseline, BenchmarkVisionText, ToMiPromptEngineering, MMToMQAPromptEngineering, MATHBenchmark
from src import (
                AssessmentTopK,
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
                 ExtendedTomLocGPT4,
                 MDLocDataset,
                 get_masked_ktop,
                 get_masked_kbot,
                 get_masked_middle,
                 get_masked_random)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_func = args.model_func
    model_type = "LLM"
    cache = args.cache

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
    avg_stim = AverageTaskStimuli(benchmark, llm, device)
    mean_imputation = MeanImputation(avg_stim.avg_activation)
    assess = Assessmentv4(import_model=llm,
                        ablation=mean_imputation,
                        device=device)
    percentages = np.concatenate(([0, 0.001, 0.002, 0.003], np.linspace(0, 0.025, 26)[::2][2:]))
    bot_predictions = []
    mid_predictions = []
    top_predictions = []
    for idx, percentage in enumerate(percentages):
        mask_bot = get_masked_kbot(classic_loc, percentage)
        mask_mid = get_masked_middle(classic_loc, percentage)
        mask_top = get_masked_ktop(classic_loc, percentage)
        res_bot = assess.experiment(benchmark, mask_bot, num_random=0, no_lesion=False)
        res_mid = assess.experiment(benchmark, mask_mid, num_random=0, no_lesion=False)
        res_top = assess.experiment(benchmark, mask_top, num_random=0, no_lesion=False)
        bot_predictions.append(res_bot["predict_ablate_top"].tolist())
        mid_predictions.append(res_mid["predict_ablate_top"].tolist())
        top_predictions.append(res_top["predict_ablate_top"].tolist())

    model_name = model_checkpoint.split("/")[-1]
    data_dict = {
        "benchmark": benchmark.data.to_dict(orient="records"),
        "bot_predictions": bot_predictions,
        "mid_predictions": mid_predictions,
        "top_predictions": top_predictions,
        "percentages": percentages.tolist()
    }
    model_name = model_checkpoint.split("/")[-1]
    json_file = os.path.join("Result", "PERCENTAGES_ANALYSIS", "MeanImputation-09-02-2025", model_name, f"{type(benchmark).__name__}_MeanImputation.json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"MeanImputation on {model_name} is done!")
    

if __name__ == "__main__":
    main()