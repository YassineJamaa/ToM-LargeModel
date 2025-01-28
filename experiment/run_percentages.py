import os
import torch
from dotenv import load_dotenv
import sys
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoProcessor,
                          LlavaNextVideoForConditionalGeneration,
                          )

from benchmark import ToMiPromptEngineering, MMToMQAPromptEngineering
from src import (setup_environment, 
                 ToMLocDataset, 
                 ExtendedTomLocGPT4, 
                 LayerUnits, 
                 LocImportantUnits, 
                 Assessmentv4, 
                 ZeroingAblation,
                 ImportModel)
from analysis import experiment, set_model, assess_percentages
from hardware.device_manager import setup_device

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_type", type=str, default="LLM", help="Model type: VLM or LLM")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--benchmark", type=str, default="ToMi", help="Benchmark that you would go into")
    parser.add_argument("--test", type=bool, default=False, help="Subset size for benchmarks (default: the whole dataset).")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    parser.add_argument("--random_num", type=int, default=10, help="Number of Random mask considering")
    return parser.parse_args()

def initialize(import_model: ImportModel, device: str):
    # Set Classic Localizer
    classic_tom = ToMLocDataset()
    classic_units = LayerUnits(import_model = import_model, localizer = classic_tom, device = device)
    classic_loc = LocImportantUnits("classic", classic_units.data_activation)
    
    # Set Extended Localizer
    extended_tom = ExtendedTomLocGPT4()
    extended_units = LayerUnits(import_model = import_model, localizer = extended_tom, device = device)
    extended_loc = LocImportantUnits("extended", extended_units.data_activation)

    ablation = ZeroingAblation() # Tune ablation
    assess = Assessmentv4(import_model=import_model,
                        ablation=ablation,
                        device=device)
    return classic_loc, extended_loc, ablation, assess

def main():
    # Parse command-line
    args = parse_arguments()
    model_type = args.model_type
    model_checkpoint = args.model_name
    model_func = args.model_func
    random_num = args.random_num
    test = args.test
    cache = args.cache
    benchmark_name = args.benchmark

    # Set up the environment & device
    token, cache_dir = setup_environment(cache=cache)
    device, device_ids = setup_device()

    # Set Model
    model_args = {
        "model_type": model_type,
        "model_checkpoint": model_checkpoint,
        "token_hf": token,
        "cache_dir": cache_dir,
        "model_func": model_func,
    }
    import_model = set_model(**model_args)
    model_name = model_checkpoint.split("/")[-1] if not test else f"TEST_{model_checkpoint.split("/")[-1]}" # Extract Model name
    percentages = np.linspace(0, 0.02, 21) if not test else np.linspace(0.0, 0.01, 3) # Set percentages interval
    
    # Tune Benchmark
    if benchmark_name=="ToMi":
        benchmark = ToMiPromptEngineering()
    elif benchmark_name =="MMToMQA":
        benchmark = MMToMQAPromptEngineering()

    # Initialize Experiment
    classic_loc, extended_loc, ablation, assess = initialize(import_model, device)

    assess_percentages(assess=assess, benchmark=benchmark, model_name=model_name, percentages=percentages, loc=classic_loc, nums=random_num)
    assess_percentages(assess=assess, benchmark=benchmark, model_name=model_name, percentages=percentages, loc=extended_loc, nums=0)

if __name__ == "__main__":
    main()









