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

from benchmark import MMStarBenchmark, MathVistaBenchmark
from src import (
                Assessmentv6,
                AssessmentTopK,
                setup_environment,
                 ImportModel,
                 load_chat_template,
                 LayerUnits,
                 LocImportantUnits,
                 MDLocDataset,
                 ZeroingAblation,
                 ToMLocDataset,
                 LangLocDataset,
                 AverageTaskStimuli,
                 MeanImputation,
                 ExtendedTomLocGPT4,
                 get_masked_kbot,
                 get_masked_ktop,
                 get_masked_middle,)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="LlavaNextVideoForConditionalGeneration", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_func = args.model_func
    model_type = "VLM"
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
    vlm = set_model(**model_args)

    model_name = model_checkpoint.split("/")[-1]

    task = MDLocDataset()
    units = LayerUnits(import_model = vlm, localizer = task, device = device)
    loc = LocImportantUnits("MD", units.data_activation)
    # benchmark = MMStarBenchmark(subset=10)
    # avg_stim = AverageTaskStimuli(benchmark, vlm, device)
    # mean_imputation = MeanImputation(avg_stim.avg_activation)
    zeroing = ZeroingAblation()

    folder = os.path.join("Result_VLM", "EXPERIMENT_RESULT", model_name, "No_lesion")
    os.makedirs(os.path.dirname(folder), exist_ok=True)

    mmstar = MMStarBenchmark(subset=None)
    mathvista = MathVistaBenchmark(subset=None)
    benchmarks = [mathvista]

    assess = Assessmentv6(import_model=vlm,
                        ablation=zeroing,
                        device=device)

    print(f"Benchmark Analysis for {model_name}...")
    percentage=0.0 # No Lesion: pct=0.0%

    for benchmark in benchmarks:
        nomask = get_masked_kbot(loc, percentage)
        res = assess.experiment(benchmark, nomask, num_random=0, no_lesion=False)
        prediction = res["predict_ablate_top"].tolist()

        # print Chance Threshold & Accuracy
        print(f"  >>{type(benchmark).__name__} Analysis...")
        chance_threshold = benchmark.data["answer_letter"].value_counts(normalize=True).max()
        accuracy = (res["predict_ablate_top"] == res["answer_letter"]).mean()
        print(f"  >> Chance Threshold = {chance_threshold:.2%}. Accuracy Model without Lesion= {accuracy:.2%}")
        
        # Save json file
        json_file = os.path.join(folder, f"{type(benchmark).__name__}_nolesion.json")
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        columns_to_drop = ['image', 'raw_byte_img', "decoded_image"]
        dataframe = benchmark.data.copy()
        dataframe = dataframe.drop(columns=[col for col in columns_to_drop if col in dataframe.columns])
        data_dict = {
            "benchmark": dataframe.to_dict(orient="records"),
            "predictions": [prediction],
            "percentages": [percentage]
        }
        with open(json_file, "w") as f:
            json.dump(data_dict, f, indent=4)
        
        print(f"Assessment Done!")

if __name__ == "__main__":
    main()