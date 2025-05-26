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
    parser.add_argument("--localizer", type=str, default="MD", help="Either: MD or ToM or language")
    parser.add_argument("--benchmark_name", type=str, default="ToMi", choices=["FanToM", "OpenToM", "ToMi", "variant_FanToM", "MATH"], help="Either: MD or ToM")
    return parser.parse_args()

benchmark_config = {
    "ToMi": ToMiPromptEngineering,
    "MATH": MATHBenchmark,
    "FanToM": partial(FanToMPromptEngineering, is_full=False),
    "OpenToM": OpenToMPromptEngineering
}

def saving_result(benchmark, predictions, percentages, json_file):
    # Prepare dictionary to store in JSON
    result_data = {
        "benchmark": benchmark.data.to_dict(orient="records"),
        "predictions": predictions,
        "percentages": percentages
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

    ######### Set the FRAMEWORK ####################################
    if localizer == "MD":
        loc_data = MDLocDataset()
    elif localizer == "ToM":
        loc_data = ToMLocDataset()
    elif localizer == "Language":
        loc_data = LangLocDataset()


    units = LayerUnits(import_model = llm, localizer = loc_data, device = device)
    loc = LocImportantUnits("classic", units.data_activation)
    zeroing = ZeroingAblation()
    assess = Assessmentv6(import_model=llm,
                        ablation=zeroing,
                        device=device)
    ##############################################################

    ########## Initialize #########################################
    subset= None
    if benchmark_name=="variant_FanToM":
        benchmark = benchmark_config["FanToM"](subset=subset, prompt_type="variant") # RESET TO NONE AFTER
    else:
        benchmark = benchmark_config[benchmark_name](subset=subset) # RESET TO NONE AFTER
    json_file = os.path.join("Result", "ABSOLUTE", f"BOT_MASK_{localizer}", model_name, f"{benchmark_name}_bottom_lesion.json")
    ###############################################################

    ################################################################
    ##################### MAIN SCRIPT ##############################
    ################################################################
    percentages = [0.005, 0.01]
    predictions = []
    for percentage in percentages:
        print(f"{model_name.upper()}:")
        print(f"Benchmark: {type(benchmark).__name__} analysis...")
        mask_bot = get_masked_kbot(loc, percentage)
        res_bot = assess.experiment(benchmark, mask_bot, num_random=0, no_lesion=False)
        accs = (res_bot["predict_ablate_top"].str.lower()==res_bot["answer_letter"]).mean()
        print(f"Bot-{percentage:.2%} Accuracy={accs:.2%}")
        predictions.append(res_bot["predict_ablate_top"].tolist())
    saving_result(benchmark, predictions, percentages, json_file)

    # ##################################################################
    # #################### Save loc ####################################
    # loc_file = os.path.join("Result", "CLAIM2", f"BOT_MASK_{localizer}", model_name, "units", f"{localizer}_loc.pkl")
    # os.makedirs(os.path.dirname(loc_file), exist_ok=True)
    # with open(loc_file, "wb") as f:
    #     pickle.dump(loc.data_activation, f)
    # print(f"Saving LocImportantUnits variable.")
    # avg_stim = AverageTaskStimuli(benchmark, llm, device)
    # avg_file = os.path.join("Result", "CLAIM2", f"BOT_MASK_{localizer}", model_name, "units", f"{localizer}_mean_{benchmark_name}.pkl")
    # os.makedirs(os.path.dirname(avg_file), exist_ok=True)
    # with open(avg_file, "wb") as f:
    #     pickle.dump(avg_stim.avg_activation, f)
    # print(f"Saving Average Stimulus on a variable.")
    # ##################################################################

    print("Process completed.")

if __name__ == "__main__":
    main()