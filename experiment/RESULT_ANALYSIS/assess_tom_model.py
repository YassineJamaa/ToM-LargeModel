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
                 get_masked_ktop)
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
    model_name = model_checkpoint.split("/")[-1]

    # ToM Localizer
    classic_tom = ToMLocDataset()
    classic_units = LayerUnits(import_model = llm, localizer = classic_tom, device = device)
    classic_loc = LocImportantUnits("classic", classic_units.data_activation)

    # MD Localizer
    md = MDLocDataset()
    md_units = LayerUnits(import_model = llm, localizer = md, device = device)
    md_loc = LocImportantUnits("classic", md_units.data_activation)

    # Folder where to save the result
    folder = os.path.join("Result", "EXPERIMENT_RESULT", model_name, "No_lesion")
    os.makedirs(os.path.dirname(folder), exist_ok=True)

    # ToM Benchmarks
    math = MATHBenchmark()
    tomi = ToMiPromptEngineering()
    fantom = FanToMPromptEngineering(is_full=False)
    opentom = OpenToMPromptEngineering(is_full=True)
    tom_benchmarks = [tomi, fantom, opentom]
    
    # Assessment
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=llm,
                        ablation=zeroing,
                        device=device)
    
    print(f"Benchmark Analysis for {model_name}...")
    percentage=0.0 # No Lesion: pct=0.0%
    for benchmark in tom_benchmarks:
        nomask = get_masked_kbot(classic_loc, percentage)
        res = assess.experiment(benchmark, nomask, num_random=0, no_lesion=False)
        prediction = res["predict_ablate_top"].tolist()

        # print Chance Threshold & Accuracy
        print(f"  >>{type(benchmark).__name__} Analysis...")
        chance_threshold = benchmark.data["answer_letter"].value_counts(normalize=True).max()
        accuracy = (res["predict_ablate_top"].str.lower() == res["answer_letter"]).mean()
        print(f"  >> Chance Threshold = {chance_threshold:.2%}. Accuracy Model without Lesion= {accuracy:.2%}")
        
        # Save json file
        json_file = os.path.join(folder, f"short_{type(benchmark).__name__}_nolesion.json")
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        data_dict = {
            "benchmark": benchmark.data.to_dict(orient="records"),
            "predictions": [prediction],
            "percentages": [percentage]
        }
        with open(json_file, "w") as f:
            json.dump(data_dict, f, indent=4)
        
        print(f"No-Lesion Assessment Done!")

if __name__ == "__main__":
    main()
