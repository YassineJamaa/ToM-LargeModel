import os
import pandas as pd
from benchmark import (BenchmarkToMi,
                       BenchmarkFanToM,
                       BenchmarkOpenToM,
                       BenchmarkMMToMQA,
                       BenchmarkText,
                       BenchmarkBaseline)
from analysis.block import AnalysisBlock

config = {
    "benchmark_dict": {
        "ToMi": BenchmarkToMi,
        "FanToM": BenchmarkFanToM,
        "OpenToM": BenchmarkOpenToM,
        "MMToMQA": BenchmarkMMToMQA
    },
    "percentages": [0.01],
    "imputations": ["zeroing"],
    "localizer_types": ["classic"]
}

def get_result(res: pd.DataFrame):
    acc_no = (res["predict_no_ablation"]==res["answer_letter"]).mean()
    acc_top = (res["predict_ablate_top"]==res["answer_letter"]).mean()
    thresh_chance = res["answer_letter"].value_counts(normalize=True).max()
    # calculate mean of random-K% across 10 samples
    # calculate its 95% CI
    print(f">> Chance Threshold: {thresh_chance:.2%}")
    print(f">> Accuracy No Ablation (%)= {acc_no:.2%}")
    print(f">> Difference Performance accuracy(%)= {acc_top-acc_no:.2%}")

def save_results(results:pd.DataFrame, 
                 data_dir:str, 
                 benchmark_name: str, 
                 percentage: float, 
                 imputation: str, 
                 localizer_type: str):
    """Save results to a CSV file."""
    data_file = f"{data_dir}/{benchmark_name}_{percentage}_{imputation}_{localizer_type}.csv"
    results.to_csv(data_file, index=False)

def run_benchmark(block: AnalysisBlock, 
                  benchmark: BenchmarkBaseline, 
                  data_dir: str, 
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
                save_results(results, data_dir, benchmark_name, percentage, imputation, localizer_type)
                get_result(results)

def script(model_args: dict, device: str, benchmark_name: str, directory: str, subset: int = 10):
    """Script for VLM and LLM."""
    benchmark_fn = config["benchmark_dict"][benchmark_name]
    benchmark = benchmark_fn(subset=subset)
    block = AnalysisBlock(model_args, device)
    data_dir = os.path.join(directory, "data")

    # Determine model type and run the appropriate benchmarks
    if block.import_model.model_type == "LLM" or isinstance(benchmark, BenchmarkText):
        run_benchmark(block, benchmark, data_dir, benchmark_name, "text-only")
    elif block.import_model.model_type == "VLM":
        for modality in ["vision-text", "text-only"]:
            run_benchmark(block, benchmark, data_dir, benchmark_name, modality)

    
        

    
                    




