import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from typing import Optional
import argparse
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (ImportLLM, 
                 LayersUnitsLLM, 
                 LocImportantUnits,
                 ToMLocDataset,
                 AssessBenchmark,)

from benchmark import BenchmarkToMi, BenchmarkOpenToM

def setup_environment():
    """Load environment variables and configure device."""
    load_dotenv()  # Ensure .env is in the project root
    hf_access_token = os.getenv("HF_ACCESS_TOKEN")
    cache_dir = os.getenv("CACHE_DIR")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    return hf_access_token, cache_dir, device


def load_model_and_tokenizer(checkpoint: str, cache_dir: Optional[str], hf_access_token: Optional[str], device):
    """Load the model and tokenizer onto the specified device."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir, token=hf_access_token)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir, token=hf_access_token).to(device)
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
    print(f"Allocated GPU memory: {allocated_memory:.2f} GB (actively used by tensors)")
    print(f"Reserved GPU memory: {reserved_memory:.2f} GB (reserved by PyTorch but not actively used)")
    return model, tokenizer

def ensure_directory_exists(path: str):
    """Ensure the directory exists; create it if it does not."""
    os.makedirs(path, exist_ok=True)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--bench_list", nargs='+', default=["OpenToM"], help="Benchmark that you would go into")
    parser.add_argument("--opentom", type=str, default=None, help="Choose between simple or complex story")
    parser.add_argument("--pct", type=float, required=True, help="Top percentage for ablation.")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for benchmarks (default: 10).")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set up the environment
    hf_access_token, cache_dir, device = setup_environment()

    # Extract arguments
    checkpoint = args.model
    pct = args.pct
    subset = args.subset
    benchmark_list = args.bench_list
    story_opentom = args.opentom

    # Extract the model name from the checkpoint path
    model_name = checkpoint.split("/")[-1]

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint, cache_dir, hf_access_token, device)

    # Assess OpenToM Benchmark
    tom_data = ToMLocDataset()
    llm = ImportLLM(model, tokenizer)
    units = LayersUnitsLLM(llm, tom_data)
    loc_units = LocImportantUnits(checkpoint, units.data_activation)

    # Output OpenToM
    output_dir = os.path.join(".", "Output", model_name)
    ensure_directory_exists(output_dir)

    # Create a checkpoint for large Benchmark
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoint", model_name)
    ensure_directory_exists(checkpoint_dir)

    # Assessment Class
    bn_assess = AssessBenchmark(llm, loc_units)

    # BENCHMARK -> OpenToM
    if "OpenToM" in benchmark_list:
        if story_opentom:
            checkpoint_file_opentom = f"{checkpoint_dir}/tmp_Variant_OpenToM_{model_name.replace('.', '_')}_{str(int(pct*100)).replace(".", "_")}.csv"
            output_path_opentom = os.path.join(output_dir, f"Variant_OpenToM_{model_name.replace('.', '_')}-pct-{str(int(pct*100)).replace(".", "_")}.csv")
        else:
            checkpoint_file_opentom = f"{checkpoint_dir}/tmp_OpenToM{model_name.replace('.', '_')}_{str(int(pct*100)).replace(".", "_")}.csv"
            output_path_opentom = os.path.join(output_dir, f"OpenToM_{model_name.replace('.', '_')}-pct-{str(int(pct*100)).replace(".", "_")}.csv")
        # Assess ToMi Benchmark
        bench_open = BenchmarkOpenToM(story=story_opentom, order="first_order", subset=subset)
        opentom_res = bn_assess.experiment(bench_open, check_path=checkpoint_file_opentom, batch_size=10, pct=pct)
        opentom_res.to_csv(output_path_opentom, index=False)
        print(f"Results saved to {output_path_opentom}")

    # Output ToMi results
    if "ToMi" in benchmark_list:
        bench_tomi = BenchmarkToMi(subset=subset)
        tomi_res = bn_assess.experiment(bench_tomi, pct=pct)
        output_path_tomi = os.path.join(output_dir, f"ToMi_{model_name.replace('.', '_')}-pct-{int(pct*100)}.csv")
        tomi_res.to_csv(output_path_tomi, index=False)
        print(f"Results saved to {output_path_tomi}")


if __name__ == "__main__":
    main()