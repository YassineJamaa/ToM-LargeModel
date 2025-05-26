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

from benchmark import BenchmarkMMToMQA, BenchmarkToMi, BenchmarkOpenToM, BenchmarkFanToM, BenchmarkBaseline, BenchmarkVisionText, ToMiPromptEngineering, MMToMQAPromptEngineering, MATHBenchmark, OpenToMPromptEngineering
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
                 get_masked_ktop)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--localizer", type=str, default="ToM", help="Either: MD or ToM")
    parser.add_argument("--benchmark_name", type=str, default="FanToM", choices=["FanToM", "MATH", "OpenToM"], help="Either: MD or ToM")
    return parser.parse_args()

benchmark_config = {
    "MATH": MATHBenchmark,
    "FanToM": BenchmarkFanToM,
    "OpenToM": OpenToMPromptEngineering
}

def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
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

    units = LayerUnits(import_model = llm, localizer = loc_data, device = device)
    loc = LocImportantUnits("classic", units.data_activation)

    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=llm,
                        ablation=zeroing,
                        device=device)
    
    ################################# BENCHMARK #####################
    benchmark = benchmark_config[benchmark_name](subset=None)
    ###################################################################
    
    percentages = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.25])
    # bot_predictions = []
    top_predictions = []
    dataframe = benchmark.data.to_dict(orient="records")
    for idx, percentage in enumerate(percentages):
        # mask_bot = get_masked_kbot(loc, percentage)
        mask_top = get_masked_ktop(loc, percentage)
        # res_bot = assess.experiment(benchmark, mask_bot, num_random=0, no_lesion=False)
        res_top = assess.experiment(benchmark, mask_top, num_random=0, no_lesion=False)
        # bot_predictions.append(res_bot["predict_ablate_top"].tolist())
        top_predictions.append(res_top["predict_ablate_top"].tolist())
    
    model_name = model_checkpoint.split("/")[-1]
    json_file= os.path.join("Result", "TOP_EXPERIMENT_FANTOM_MD", model_name, f"{type(benchmark).__name__}_top.json") #CHANGE
    data_dict = {
        "percentages": percentages.tolist(),
        "benchmark": dataframe,
        # "bot_predictions": bot_predictions,
        "top_predictions": top_predictions
    }
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"Top Analysis done.")

if __name__ == "__main__":
    main()