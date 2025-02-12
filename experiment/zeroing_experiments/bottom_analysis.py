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
                 get_masked_kbot,
                 get_masked_ktop)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--localizer", type=str, default="MD", help="Either: MD or ToM")
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_func = args.model_func
    model_type = "LLM"
    cache = args.cache
    localizer = args.localizer

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

    localizer = "MD"
    if localizer == "MD":
        context = (
            "This is a math problem."
            "Review the question carefully and select the best answer from the given options."
            " After 'Answer:', respond with only one choice: a, b, c, or d."
        )
        loc_data = MDLocDataset()
        benchmark = MATHBenchmark(subset=None, context=context) #CHange
    elif localizer == "ToM":
        loc_data = ToMLocDataset()
        benchmark = ToMiPromptEngineering

    units = LayerUnits(import_model = llm, localizer = loc_data, device = device)
    loc = LocImportantUnits("classic", units.data_activation)

    zeroing = ZeroingAblation()
    target_chars = {"a", "b", "c", "d"}
    assess = AssessmentTopK(import_model=llm,
                        ablation=zeroing,
                        target_chars=target_chars,
                        device=device)
    
    percentages = np.linspace(0.0, 0.025, 26)[::2] # Change
    bot_predictions = []
    top_predictions = []
    dataframe = benchmark.data.to_dict(orient="records")
    for idx, percentage in enumerate(percentages):
        mask_bot = get_masked_kbot(loc, percentage)
        mask_top = get_masked_ktop(loc, percentage)
        res_bot = assess.experiment(benchmark, mask_bot, num_random=0, no_lesion=False)
        res_top = assess.experiment(benchmark, mask_top, num_random=0, no_lesion=False)
        bot_predictions.append(res_bot["predict_ablate_top"].tolist())
        top_predictions.append(res_top["predict_ablate_top"].tolist())
    
    model_name = model_checkpoint.split("/")[-1]
    json_file= os.path.join("Result", f"TOPK_{localizer}_BOTTOM_ANALYSIS", model_name, f"{type(benchmark).__name__}_bottom.json") #CHANGE
    data_dict = {
        "benchmark": dataframe,
        "bot_predictions": bot_predictions,
        "top_predictions": top_predictions
    }
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"Bottom Analysis done.")

if __name__ == "__main__":
    main()