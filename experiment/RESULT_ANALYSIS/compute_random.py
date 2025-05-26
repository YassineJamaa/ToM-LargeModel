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

from benchmark import BenchmarkMMToMQA, BenchmarkToMi, BenchmarkOpenToM, BenchmarkFanToM, BenchmarkBaseline, BenchmarkVisionText, ToMiPromptEngineering, MMToMQAPromptEngineering, MATHBenchmark, OpenToMPromptEngineering, FanToMPromptEngineering
from src import (
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
    parser.add_argument("--benchmark", type=str, choices=["ToMi", "full_OpenToM", "FanToM", "MATH"], default="FanToM", help="ToM/MultipleDemand Benchmark")
    parser.add_argument("--lesion", type=str, choices=["Zeroing", "MeanImputation"], default="Zeroing", help="Lesion Method")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    return parser.parse_args()


benchmark_config = {
    "ToMi": ToMiPromptEngineering,
    "full_OpenToM": OpenToMPromptEngineering,
    "FanToM": FanToMPromptEngineering,
}

def print_result(res):
    print(f"  >> Acc. = {(res["predict_ablate_top"] == res["answer_letter"]).mean():.2%}")

def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_name = model_checkpoint.split("/")[-1]
    model_func = args.model_func
    model_type = "LLM"
    cache = args.cache
    benchmark_name = args.benchmark
    lesion_method = args.lesion

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

    contrast = ToMLocDataset() if benchmark_name != "MATH" else MDLocDataset()
    units = LayerUnits(import_model = llm, localizer = contrast, device = device)
    loc = LocImportantUnits("localizer", units.data_activation)

    ################ BENCHMARK ##################################
    benchmark = benchmark_config[benchmark_name]()
    ############################################################
    
    ################## LESION METHOD ########################
    if lesion_method=="MeanImputation":
        avg_stim = AverageTaskStimuli(benchmark, llm, device)
        lesion = MeanImputation(avg_stim.avg_activation)
    else:
        lesion = ZeroingAblation()
    ######################################################


    assess = Assessmentv6(import_model=llm,
                        ablation=lesion,
                        device=device)
    

    ################# Setting ###############################################
    percentages = np.array([0.05])
    random_predictions = []
    n_random = 50
    ########################################################################

    ############## Print the config ########################################
    print(f"Model: {model_name}\n >> Setup:\n    >> Benchmark: {benchmark_name}\n    >> Localizer: {type(contrast).__name__}\n    >> Lesion Method: {lesion_method}")
    ########################################################################


    for idx, percentage in enumerate(percentages):
        print(f"Percentage = {percentage:.2%}")
        predictions_template = []
        for seed in range(n_random):
            mask_random = get_masked_random(loc, percentage, seed)
            res_random = assess.experiment(benchmark, mask_random, num_random=0, no_lesion=False)
            print_result(res_random)
            predictions_template.append(res_random["predict_ablate_top"].tolist())
        random_predictions.append(predictions_template)
    
    model_name = model_checkpoint.split("/")[-1]
    json_file= os.path.join("Result", "EXPERIMENT_RESULT", model_name, "Lesion", benchmark_name, f"random_{benchmark_name}_{lesion_method}.json")
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
