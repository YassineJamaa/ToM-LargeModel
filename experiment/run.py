import os
import torch
from dotenv import load_dotenv
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoProcessor,
                          LlavaNextVideoForConditionalGeneration,
                          )

from src import setup_environment
from analysis import experiment
from hardware.device_manager import setup_device


def percentage_folder(percentage:float):
    # percentage folder
    if (percentage * 100).is_integer():
        pct_folder = f"top-{percentage*100:.0f}%"  # No decimal places
    else:
        pct_folder = f"top-{percentage*100:.1f}%"   # One decimal place
    return pct_folder

def build_model_directory(model_type: str, model_name: str):
    output_dir = os.path.join(".", "Result", model_type, model_name)
    for name_dir in ["data", "json_results"]:
        dir = os.path.join(output_dir, name_dir)
        os.makedirs(dir, exist_ok=True)
    return output_dir  

# ["ToMi", "FanToM", "Variant_FanToM", "OpenToM", "Variant_OpenToM", "MMToMQA"],  

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_type", type=str, default="LLM", help="Model type: VLM or LLM")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--benchmarks", nargs='+', default=["ToMi"], help="Benchmark that you would go into")
    parser.add_argument("--test", type=bool, default=False, help="Subset size for benchmarks (default: the whole dataset).")
    return parser.parse_args()

def main():
    # Parse command-line
    args = parse_arguments()
    model_type = args.model_type
    model_checkpoint = args.model_name
    model_func = args.model_func
    benchmarks = args.benchmarks
    test = args.test

    # Set up the environment
    token, cache_dir = setup_environment()

    # Set up the device
    device, device_ids = setup_device()

    # Extract the model name from the model checkpoint
    model_name = model_checkpoint.split("/")[-1]

    # Build model directory
    result_dir = build_model_directory(model_type, model_name)

    # Set "ImportModel"
    model_args = {
        "model_type": model_type,
        "model_checkpoint": model_checkpoint,
        "token_hf": token,
        "cache_dir": cache_dir,
        "model_func": model_func,
    }
    subset = 10 if test else None
    experiment(model_args, device, benchmarks, result_dir, subset)
    


if __name__ == "__main__":
    main()

