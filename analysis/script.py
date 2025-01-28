import os
import pandas as pd
from scipy.stats import t
import numpy as np
import json
from functools import partial
from benchmark import (BenchmarkToMi,
                       BenchmarkFanToM,
                       BenchmarkOpenToM,
                       BenchmarkMMToMQA,
                       BenchmarkText,
                       BenchmarkBaseline)
from src import (
    ZeroingAblation,
)
from analysis.block import AnalysisBlock
from torch.utils.data import Dataset
from analysis.utils import compute_t_confidence_interval

config = {
    "benchmark_dict": {
        "ToMi": BenchmarkToMi,
        "FanToM": BenchmarkFanToM,
        "Variant_FanToM": partial(BenchmarkFanToM, is_full=False),
        "OpenToM": BenchmarkOpenToM,
        "Variant_OpenToM": partial(BenchmarkOpenToM, is_full=False),
        "MMToMQA": BenchmarkMMToMQA,
    },
    "benchmark_group":{
        "ToMi": None,
        "FanToM": None,
        "Variant_FanToM": None,
        "OpenToM": "Qtype",
        "Variant_OpenToM": "Qtype",
        "MMToMQA": "Qtype",
    },
    "percentages": [0.01],
    "imputations": ["zeroing"],
    "localizer_types": ["classic", "extended"]
}

def get_result(res: pd.DataFrame):
    acc_no = (res["predict_no_ablation"]==res["answer_letter"]).mean()
    acc_top = (res["predict_ablate_top"]==res["answer_letter"]).mean()
    thresh_chance = res["answer_letter"].value_counts(normalize=True).max()
    mean_random_array = np.array([
        (res[col] == res['answer_letter']).mean() 
        for col in res.filter(like='predict_ablate_random').columns
    ])
    acc_random, lower_random, upper_random = compute_t_confidence_interval(mean_random_array)

    # Check if the test is successful
    is_successful = (thresh_chance < acc_no) and (acc_top - acc_no) < 0 and (acc_top < lower_random)

    # Print success or failure information
    if is_successful:
        print("    ✔️  Test Result: ✅ SUCCESS (Performance meets expectations)")
    else:
        print("    ⚠️  Test Result: ❌ FAILURE (Performance does not meet expectations)")

    print(f"        >> Chance Threshold(%)= {thresh_chance:.2%}")
    print(f"        >> Accuracy No Ablation(%)= {acc_no:.2%}")
    print(f"        >> Top-K%: Difference Performance accuracy(%)= {acc_top-acc_no:.2%}")
    print(f"        >> Random-K%: Difference Performance accuracy(%)= {acc_random-acc_no:.2%} (95% CI: {lower_random-acc_no:.2%} to {upper_random-acc_no:.2%})")
    acc_dict = {
        "chance threshold": thresh_chance,
        "no ablation": acc_no,
        "top ablation": acc_top,
        "random ablation": acc_random,
        "lower random": lower_random,
        "upper random": upper_random
    }
    return acc_dict

def get_results_benchmarks(results: pd.DataFrame,
                           benchmark_name: str):
    acc_dict = {}
    if config["benchmark_group"][benchmark_name]:
        for name, sub_df in results.groupby(config["benchmark_group"][benchmark_name]):
            # Apply subgroup filter only for the "MMToMQA" benchmark
            if benchmark_name == "MMToMQA" and name not in ['False belief', 'Goal inference false belief']:
                continue  # Skip subgroups not in [1.2, 2.2]
            print(f">> Question Type: {name}")
            acc_dict[name] = get_result(sub_df)
    else:
        acc_dict = get_result(results)
    return acc_dict
            

def run_analysis(results: pd.DataFrame,
                 benchmark_name: str,
                 percentage: int,
                 data_dir: str,
                 json_dir: str,
                 imputation: ZeroingAblation,
                 localizer_type: Dataset):
    save_data_results(results, data_dir, benchmark_name, percentage, imputation, localizer_type)
    acc_dict = get_results_benchmarks(results, benchmark_name)
    save_json_results(acc_dict, json_dir, benchmark_name, percentage, imputation, localizer_type)



def save_data_results(results:pd.DataFrame, 
                 data_dir:str, 
                 benchmark_name: str, 
                 percentage: float, 
                 imputation: str, 
                 localizer_type: str):
    """Save results to a CSV file."""
    data_file = f"{data_dir}/{benchmark_name}_{percentage}_{imputation}_{localizer_type}.csv"
    results.to_csv(data_file, index=False)

def save_json_results(acc_dict: dict,
                      json_dir: str,
                      benchmark_name: str, 
                      percentage: float, 
                      imputation: str, 
                      localizer_type: str):
    """ Save dict results in json file """
    acc_file = f"{json_dir}/{benchmark_name}_{percentage}_{imputation}_{localizer_type}.json"
    try:
        # Write the dictionary to the JSON file
        with open(acc_file, 'w') as json_file:
            json.dump(acc_dict, json_file, indent=4)
        # print(f"Results saved successfully to {acc_file}")
    except Exception as e:
        print(f"Error saving results to {acc_file}: {e}")


def run_benchmark(block: AnalysisBlock, 
                  benchmark: BenchmarkBaseline, 
                  data_dir: str,
                  json_dir: str, 
                  benchmark_name:str, 
                  modality: str):
    """Run benchmarks and save results based on the configuration."""
    for percentage in config["percentages"]:
        for imputation in config["imputations"]:
            for localizer_type in config["localizer_types"]:
                results = block.run(
                    benchmark=benchmark,
                    imputation=imputation,
                    percentage=percentage,
                    kind=localizer_type,
                    modality=modality
                )
                print(f"MODEL NAME: {block.model_name}")
                print(f"LOCALIZER TYPE: {localizer_type}")
                print(f"BENCHMARK: {benchmark_name}\nPERCENTAGE={percentage:.2%} & IMPUTATION={imputation}")
                run_analysis(results=results,
                            benchmark_name=benchmark_name,
                            percentage=percentage,
                            data_dir=data_dir,
                            json_dir=json_dir,
                            imputation=imputation,
                            localizer_type=localizer_type)
                

def script(block: AnalysisBlock, device: str, benchmark_name: str, directory: str, subset: int = 10, save_layers: bool=False):
    """Script for VLM and LLM."""
    benchmark_fn = config["benchmark_dict"][benchmark_name]
    benchmark = benchmark_fn(subset=subset)

    # Set th Mean Imputation
    block.set_mean_imputation(benchmark)

    # Save the layers informations and masks info from the model
    if save_layers:
        block.save_layers_dices(directory, percentage=0.01)

    # Set the data and json directory
    data_dir = os.path.join(directory, "data", benchmark_name)
    json_dir = os.path.join(directory, "json_results", benchmark_name)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)   

    # Determine model type and run the appropriate benchmarks
    if block.import_model.model_type == "LLM" or isinstance(benchmark, BenchmarkText):
        run_benchmark(block, benchmark, data_dir, json_dir, benchmark_name, "text-only")
    elif block.import_model.model_type == "VLM":
        for modality in ["vision-text", "text-only"]:
            run_benchmark(block, benchmark, data_dir, json_dir, benchmark_name, modality)

def experiment(model_args: dict, device: str, benchmarks:list, directory: str, subset: int = 10):
    block = AnalysisBlock(model_args, device)
    flag = True
    for benchmark_name in benchmarks:
        if flag:
            script(block, device, benchmark_name, directory, subset, flag)
            flag = False
        else:
            script(block, device, benchmark_name, directory, subset)



    
        

    
                    




