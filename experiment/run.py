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
                 ExtendedTomLocGPT4,
                 AssessBenchmark,
                 AverageTaskStimuli,
                 ZeroingAblation,
                 MeanImputation)

from benchmark import BenchmarkToMi, BenchmarkOpenToM, BenchmarkFanToM, BenchmarkBaseline

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
    parser.add_argument("--pct", type=float, required=True, help="Top percentage for ablation.")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for benchmarks (default: 10).")
    parser.add_argument("--batch", type=int, default=10, help="Batch size (default: 10).")
    parser.add_argument("--extended_loc", type=str, default=False, help="Choose between the classical localizer or extended one.")
    parser.add_argument("--imput", type=str, default=None, help="Choose between zeroing or mean imputation.")
    return parser.parse_args()

def delete_checkpoint_file(file_path):
    """
    Deletes the specified checkpoint file if it exists.
    
    Args:
        file_path (str): Path to the file to be deleted.
    """
    try:
        os.remove(file_path)
        print(f"Checkpoint file {file_path} has been deleted.")
    except FileNotFoundError:
        print(f"Checkpoint file {file_path} not found for deletion.")
    except Exception as e:
        print(f"An error occurred while deleting the file {file_path}: {e}")

def ablation_task(llm:ImportLLM, 
                  bench: BenchmarkBaseline, 
                  loc_units: LocImportantUnits, 
                  pct: float, 
                  batch_size: int,
                  check_path: Optional[str] = None,
                  imput: Optional[str] = None):
    if imput is None:
        ablat = ZeroingAblation()
    elif imput == "mean":
        avg = AverageTaskStimuli(bench, llm)
        ablat = MeanImputation(avg.avg_activation)

    assess = AssessBenchmark(llm, loc_units, ablat)
    return assess.experiment(bench, pct=pct, batch_size=batch_size, check_path=check_path)

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set up the environment
    hf_token, cache_dir, device = setup_environment()

    # Extract arguments
    checkpoint = args.model
    pct = args.pct
    subset = args.subset
    benchmark_list = args.bench_list
    batch_size = args.batch
    extended_loc = args.extended_loc
    imput = args.imput

    # Imputation Name
    if imput is None:
        imput_name = "zeroing"
    elif imput == "mean":
        imput_name = "mean_imput"
    else:
        raise ValueError(f"Invalid imput value: {imput}. Accepted values are None or 'mean'.")

    # percentage folder
    if (pct * 100).is_integer():
        pct_folder = f"top-{pct*100:.0f}%"  # No decimal places
    else:
        pct_folder = f"top-{pct*100:.1f}%"   # One decimal place

    # Extract the model name from the checkpoint path
    model_name = checkpoint.split("/")[-1]

    # Load model and tokenizer and Import LLM
    model, tokenizer = load_model_and_tokenizer(checkpoint = checkpoint, cache_dir=cache_dir, hf_access_token=hf_token, device=device)
    llm = ImportLLM(model, tokenizer)

    # Get the right localizer
    if extended_loc:
        # Extended Localizer
        localizer_name = "ExtendedTomLocGPT4"
        extended_tom_data = ExtendedTomLocGPT4()
        units = LayersUnitsLLM(llm, extended_tom_data)

    else:
        # Classic Localizer
        localizer_name = "ClassicTomLoc"
        tom_data = ToMLocDataset()
        units = LayersUnitsLLM(llm, tom_data)

    # Localize Important Units
    loc_units = LocImportantUnits(checkpoint, units.data_activation)

    # Output OpenToM
    output_dir = os.path.join(".", "Output", localizer_name, pct_folder, model_name)
    ensure_directory_exists(output_dir)

    # Create a checkpoint for large Benchmark
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoint", localizer_name, pct_folder, model_name)
    ensure_directory_exists(checkpoint_dir)

    # BENCHMARK Assessment
    if "FanToM" in benchmark_list:
        # FanToM Benchmark
        checkpoint_file_fantom = f"{checkpoint_dir}/tmp_FanToM_{model_name.replace('.', '_')}_{imput_name}.csv"
        output_path_fantom = os.path.join(output_dir, f"FanToM_{model_name.replace('.', '_')}_{imput_name}.csv")
        
        # Assess FanToM Benchmark
        bench_fantom = BenchmarkFanToM(subset=subset)
        fantom_res = ablation_task(llm=llm,
                                   loc_units=loc_units,
                                   bench=bench_fantom, 
                                   check_path=checkpoint_file_fantom, 
                                   batch_size=batch_size, 
                                   pct=pct)
        
        # Save results to output directory
        fantom_res.to_csv(output_path_fantom, index=False)
        print(f"Results saved to {output_path_fantom}")

        # Call the function to delete the checkpoint file
        delete_checkpoint_file(checkpoint_file_fantom)
    
    if "Variant_FanToM" in benchmark_list:
        checkpoint_file_var_fantom = f"{checkpoint_dir}/tmp_Variant_FanToM_{model_name.replace('.', '_')}_{imput_name}.csv"
        output_path_var_fantom = os.path.join(output_dir, f"Variant_FanToM_{model_name.replace('.', '_')}_{imput_name}.csv")
        
        # Assess simplifier version of FanToM Benchmark
        bench_fantom = BenchmarkFanToM(is_full=False, subset=subset)
        fantom_res = ablation_task(llm=llm,
                                   loc_units=loc_units,
                                   bench=bench_fantom, 
                                   check_path=checkpoint_file_var_fantom, 
                                   batch_size=batch_size, 
                                   pct=pct)

        # Save results to output directory
        fantom_res.to_csv(output_path_var_fantom, index=False)
        print(f"Results saved to {output_path_var_fantom}")

        # Call the function to delete the checkpoint file
        delete_checkpoint_file(checkpoint_file_var_fantom)

    if "OpenToM" in benchmark_list:
        # OpenToM Benchmark
        checkpoint_file_opentom = f"{checkpoint_dir}/tmp_OpenToM_{model_name.replace('.', '_')}_{imput_name}.csv"
        output_path_opentom = os.path.join(output_dir, f"OpenToM_{model_name.replace('.', '_')}_{imput_name}.csv")
        
        # Assess OpenToM Benchmark
        bench_open = BenchmarkOpenToM(order="first_order", subset=subset)
        opentom_res = ablation_task(llm=llm,
                                   loc_units=loc_units,
                                   bench=bench_open, 
                                   check_path=checkpoint_file_opentom, 
                                   batch_size=batch_size, 
                                   pct=pct)
        
        # Save results to output directory
        opentom_res.to_csv(output_path_opentom, index=False)
        print(f"Results saved to {output_path_opentom}")

        # Call the function to delete the checkpoint file
        delete_checkpoint_file(checkpoint_file_opentom)
    
    if "Variant_OpenToM" in benchmark_list:
        checkpoint_file_var_opentom = f"{checkpoint_dir}/tmp_Variant_OpenToM_{model_name.replace('.', '_')}_{imput_name}.csv"
        output_path_var_opentom = os.path.join(output_dir, f"Variant_OpenToM_{model_name.replace('.', '_')}_{imput_name}.csv")
        
        # Assess simplifier version of OpenToM Benchmark
        bench_open = BenchmarkOpenToM(story="plot", order="first_order", subset=subset)
        opentom_res = ablation_task(llm=llm,
                                   loc_units=loc_units,
                                   bench=bench_open, 
                                   check_path=checkpoint_file_var_opentom, 
                                   batch_size=batch_size, 
                                   pct=pct)

        # Save results to output directory
        opentom_res.to_csv(output_path_var_opentom, index=False)
        print(f"Results saved to {output_path_var_opentom}")

        # Call the function to delete the checkpoint file
        delete_checkpoint_file(checkpoint_file_var_opentom)

    # Output ToMi results
    if "ToMi" in benchmark_list:
        output_path_tomi = os.path.join(output_dir, f"ToMi_{imput_name}.csv")

        # Assess ToMi Benchmark
        bench_tomi = BenchmarkToMi(subset=subset)
        tomi_res = ablation_task(llm=llm,
                                   loc_units=loc_units,
                                   bench=bench_tomi, 
                                   batch_size=batch_size, 
                                   pct=pct)

        # Save results to output directory
        tomi_res.to_csv(output_path_tomi, index=False)
        print(f"Results saved to {output_path_tomi}")

if __name__ == "__main__":
    main()