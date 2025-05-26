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
                AssessmentLogits,
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
    parser.add_argument("--benchmark_name", type=str, default="FanToM", choices=["variant_FanToM", "FanToM", "MATH", "OpenToM", "ToMi"], help="Either: MD or ToM")
    return parser.parse_args()

benchmark_config = {
    "ToMi": ToMiPromptEngineering,
    "MATH": MATHBenchmark,
    "FanToM": partial(FanToMPromptEngineering, is_full=False),
    "OpenToM": OpenToMPromptEngineering
}

# Save 
def saving_result(benchmark_no, benchmark_lesion, percentages, json_file):
    result_data = {
        "benchmark_nolesion": benchmark_no.to_dict(orient="records"),
        "benchmark_lesion": benchmark_lesion.to_dict(orient="records"),
        "percentages": percentages
    }
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    # Write to JSON file
    with open(json_file, "w") as f:
        json.dump(result_data, f, indent=4)


def get_stats_probs(res):
    probs = np.array(res["probs_ablate_top"].tolist()).reshape(-1)
    ci_probs = 1.96 * probs.std() / np.sqrt(probs.shape[0])
    probs_mean1 = probs.mean().item()
    print(f"Mean Logits: {probs_mean1}, 95% CIs: [{probs_mean1-ci_probs},{probs_mean1+ci_probs}]")

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
    assess = AssessmentLogits(import_model=llm,
                        ablation=zeroing,
                        device=device)
    
    subset = None
    if benchmark_name=="variant_FanToM":
        benchmark = benchmark_config["FanToM"](subset=subset, prompt_type="variant") # RESET TO NONE AFTER
    else:
        benchmark = benchmark_config[benchmark_name](subset=subset) # RESET TO NONE AFTER
    
    percentages = []
    # Without Lesion
    percentage = 0.0
    percentages.append(percentage)
    mask = get_masked_kbot(noabs_loc, percentage)
    res_no = assess.experiment(benchmark, mask, topK=1)
    outliers = [prediction for prediction in res_no["predict_ablate_top"].tolist() if prediction.lower() not in ["a", "b"]]
    print(f"Nb. of Outliers {len(outliers)}")
    print(f" Accuracy Diff-{percentage:.2%} Lesion: {(res_no["predict_ablate_top"].str.lower() == res_no["answer_letter"]).mean():.2%}")
    get_stats_probs(res_no)

    # Lesion (With No-Absolute)
    percentage = 0.01
    percentages.append(percentage)
    mask = get_masked_kbot(noabs_loc, percentage)
    res_lesion = assess.experiment(benchmark, mask, topK=1)
    outliers = [prediction for prediction in res_lesion["predict_ablate_top"].tolist() if prediction.lower() not in ["a", "b"]]
    print(f"Nb. of Outliers {len(outliers)}")
    print(f" Accuracy Diff-{percentage:.2%} Lesion: {(res_lesion["predict_ablate_top"].str.lower() == res_lesion["answer_letter"]).mean():.2%}")
    get_stats_probs(res_lesion)
    
    json_file = os.path.join("Result", "Final-Analysis", "No-absolute", "LogitsAnalysis", model_name, f"{benchmark_name}_logits.json")
    saving_result(res_no, res_lesion, percentages, json_file)
    print("Process completed.")

if __name__ == "__main__":
    main()
