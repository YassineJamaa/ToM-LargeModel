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
                 get_masked_random,)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForImageTextToText", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--benchmark_names", type=str, nargs='+', choices=["MathVista", "MMStar"], default=["MathVista", "MMStar"], help="List of benchmark names. Choose from: MathVista, MMStar")
    return parser.parse_args()

benchmark_config = {
    "MathVista": MathVistaBenchmark,
    "MMStar": MMStarBenchmark,
}

def save_data_json(benchmark, model_name, dataframe, percentages, predictions, n_random):
    json_file = os.path.join("Result_VLM", "VLM_LESION_MD", model_name, f"{type(benchmark).__name__}_random_lesion.json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    data_dict = {
        "percentages": percentages.tolist(),
        "benchmark": dataframe.to_dict(orient="records"),
        "random_predictions": predictions,
        "seed": list(range(n_random))
    }
    with open(json_file, "w") as f:
        json.dump(data_dict, f, indent=4)

def assessment_benchmark_random_masks(benchmark, assess, model_name, loc, percentages, n_random):
    ############ SCRIPTS ####################################################
    columns_to_drop = ['image', 'raw_byte_img', "decoded_image"]
    dataframe = benchmark.data.copy()
    dataframe = dataframe.drop(columns=[col for col in columns_to_drop if col in dataframe.columns])
    random_predictions = []
    print(f"Benchmark Bot-Analysis for {model_name}...")
    print(f"  >> {type(benchmark).__name__} Analysis...")
    for idx, percentage in enumerate(percentages):
        predictions_template = []
        for seed in range(n_random):
            mask_random = get_masked_random(loc, percentage, seed)
            res_random = assess.experiment(benchmark, mask_random, num_random=0, no_lesion=False)
            predictions_template.append(res_random["predict_ablate_top"].tolist())
        random_predictions.append(predictions_template)
    
    save_data_json(benchmark=benchmark,
                   model_name=model_name,
                   dataframe=dataframe,
                   percentages=percentages,
                   n_random = n_random,
                   predictions=random_predictions)

def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_func = args.model_func
    benchmark_names = args.benchmark_names
    cache = args.cache

    #########################################################################
    ################## SET THE VLM ##########################################
    model_type = "VLM"
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
    ##########################################################################
    ##########################################################################
    ############## SET LOCALIZER #############################################
    task = MDLocDataset()
    units = LayerUnits(import_model = vlm, localizer = task, device = device)
    loc = LocImportantUnits("MD", units.data_activation)
    ##########################################################################
    ##########################################################################
    ############# SET LESION & ASSESSMENT ####################################
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=vlm,
                        ablation=zeroing,
                        device=device)
    ##########################################################################
    ########### SET PERCENTAGES ##############################################
    percentages = np.array([0.01])
    ##########################################################################
    ##########################################################################
    ########## SET NUMBER OF RANDOM MASK #####################################
    n_random = 15
    ##########################################################################
    ##########################################################################

    for benchmark_name in benchmark_names:
        ############# SET BENCHMARK #############################################
        benchmark = benchmark_config[benchmark_name](subset=None)
        ##########################################################################
        assessment_benchmark_random_masks(
            benchmark=benchmark,
            assess=assess,
            model_name=model_name,
            loc=loc,
            percentages=percentages,
            n_random=n_random
        )
    
    print(f"Random Analysis done.")

if __name__ == "__main__":
    main()




