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
                 get_masked_random)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--percentage", type=float, default=0.002, help="A single percentage value (e.g., 0.002).")
    return parser.parse_args()



def main():
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
    
    percentages = np.array([percentage])
    random_predictions = []
    dataframe = benchmark.data.to_dict(orient="records")
    n_random = 75
    for idx, percentage in enumerate(percentages):
        predictions_template = []
        for seed in range(n_random):
            mask_random = get_masked_random(classic_loc, percentage, seed)
            res_random = assess.experiment(benchmark, mask_random, num_random=0, no_lesion=False)
            predictions_template.append(res_random["predict_ablate_top"].tolist())
        random_predictions.append(predictions_template)
    
    model_name = model_checkpoint.split("/")[-1]
    json_file= os.path.join("Result", "6.Bot_Zeroing_RANDOM_ToMi", f"RANDOM_top1_ANALYSIS", model_name, f"{type(benchmark).__name__}_randoms.json")
    data_dict = {
        "benchmark": benchmark.data.to_dict(orient="records"),
        "percentages": percentages.tolist(),
        "seeds": np.arange(n_random).tolist() ,
        "random_predictions": random_predictions,
    }
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"Random analysis Task for {model_name}: Done!")

if __name__ == "__main__":
    main()