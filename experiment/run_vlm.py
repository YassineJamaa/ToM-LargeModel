import os
import torch
from dotenv import load_dotenv
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration
import sys
from typing import Optional
import argparse
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    ImportVLM,
    LocImportantUnits,
    LayersUnitsVLM,
    ZeroingAblation,
    AssessMMToM,
    ToMLocDataset,
    ExtendedTomLocGPT4
)
from benchmark import BenchmarkMMToMQA
from hardware.device_manager import setup_device

def get_model_args(model_name, cache_dir=None, token=None):
    """
    Prepares and returns the arguments for loading a Hugging Face model with `from_pretrained`.
    Optional parameters like `cache_dir` and `token` are included only if provided (not None).
    """
    print("\n🛠 Preparing model arguments...")

    # Initialize the argument dictionary
    from_pretrained_args = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }

    # Add optional arguments if they are not None
    if cache_dir is not None:
        from_pretrained_args["cache_dir"] = cache_dir
    if token is not None:
        from_pretrained_args["token"] = token

    print("✅ Model arguments prepared")
    return from_pretrained_args

def setup_environment():
    """Load environment variables and configure device."""
    load_dotenv()  # Ensure .env is in the project root
    hf_access_token = os.getenv("HF_ACCESS_TOKEN")
    cache_dir = os.getenv("CACHE_DIR")
    return hf_access_token, cache_dir

def load_model_and_tokenizer(model_args: dict, device: str, device_ids: list):
    """Load the model and tokenizer onto the specified device."""
    if model_args["pretrained_model_name_or_path"] == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(**model_args)
        is_video = True
    elif model_args["pretrained_model_name_or_path"] == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        model = MllamaForConditionalGeneration.from_pretrained(**model_args)
        is_video = False
    else:
        raise ValueError(f"Model not recognize.")
    
    if len(device_ids)>1:
        model.tie_weights()
        model.to("auto_device")
    else:
        model.to(device)
    
    processor_args = {key:value for key, value in model_args.items() if key not in ["torch_dtype"]}
    processor = AutoProcessor.from_pretrained(**processor_args)
    tokenizer = AutoTokenizer.from_pretrained(**processor_args)
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
    print(f"Allocated GPU memory: {allocated_memory:.2f} GB (actively used by tensors)")
    print(f"Reserved GPU memory: {reserved_memory:.2f} GB (reserved by PyTorch but not actively used)")
    return model, processor, tokenizer, is_video

def ensure_directory_exists(path: str):
    """Ensure the directory exists; create it if it does not."""
    os.makedirs(path, exist_ok=True)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="Model checkpoint path.")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for benchmarks (default: 10).")
    parser.add_argument("--extended_loc", type=str, default=False, help="Choose between the classical localizer or extended one.")
    parser.add_argument("--impute", type=str, default=None, help="Choose between zeroing or mean imputation.")
    parser.add_argument("--pct", type=float, default=0.01, help="Top percentage for ablation.")
    parser.add_argument("--modality", type=str, default="multi", help="Input modality type: 'multi' for Vision-Text inputs, or 'text' for Text-Only inputs.")
    return parser.parse_args()

def ablation_task(vlm: ImportVLM,
                  bench: BenchmarkMMToMQA,
                  loc_units: LocImportantUnits,
                  pct: float,
                  is_video: bool,
                  modality: str,
                  impute: Optional[str]=None):
    if impute is None:
        print("Using Zeroing Ablation.")
        ablat = ZeroingAblation()
    
    else:
        raise ValueError(f"Invalid impute value: {impute}. Accepted value is only None.")
    
    assess = AssessMMToM(vlm, bench, loc_units, ablat, is_video)
    return assess.experiment(pct, modality)

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Extract arguments
    checkpoint = args.model
    pct = args.pct
    subset = args.subset
    impute = args.impute
    extended_loc = args.extended_loc
    modality = args.modality

    # Set up the environment & Device Set Up & Model arguments
    hf_token, cache_dir = setup_environment()
    device, device_ids = setup_device()
    model_args = get_model_args(model_name=checkpoint, cache_dir=cache_dir, token=hf_token)

    # Imputation Name
    if impute is None:
        impute_name = "zeroing"
    else:
        raise ValueError(f"Invalid impute value: {impute}. Accepted value is only None (for now).")
    
    # percentage folder
    if (pct * 100).is_integer():
        pct_folder = f"top-{pct*100:.0f}%"  # No decimal places
    else:
        pct_folder = f"top-{pct*100:.1f}%"   # One decimal place
    
    # Extract the model name from the checkpoint path
    model_name = checkpoint.split("/")[-1]

    # Load model and tokenizer and Import LLM
    model, processor, tokenizer, is_video = load_model_and_tokenizer(model_args, device, device_ids)
    vlm = ImportVLM(model, processor, tokenizer)

    # Get the right localizer
    if extended_loc:
        # Extended Localizer
        localizer_name = "ExtendedTomLocGPT4"
        extended_tom_data = ExtendedTomLocGPT4()
        units = LayersUnitsVLM(vlm, extended_tom_data)

    else:
        # Classic Localizer
        localizer_name = "ClassicTomLoc"
        tom_data = ToMLocDataset()
        units = LayersUnitsVLM(vlm, tom_data)
    
    # Localize Important Units
    loc_units = LocImportantUnits(checkpoint, units.data_activation)

    # Output MMToMQA Benchmark
    output_dir = os.path.join(".", "Output", "VLM", localizer_name, pct_folder, model_name)
    ensure_directory_exists(output_dir)

    # MMToMQA
    output_path_mmtom = os.path.join(output_dir, f"MMToMQA_{model_name.replace('.', '_')}_{impute_name}_{modality.upper()}.csv")
    bench_mmtom = BenchmarkMMToMQA(subset=subset)
    mmtom_res = ablation_task(vlm=vlm,
                              loc_units=loc_units,
                              impute=impute,
                              bench=bench_mmtom,
                              pct=pct,
                              is_video=is_video,
                              modality=modality)
    
    # Save results to output directory
    mmtom_res.to_csv(output_path_mmtom, index=False)
    print(f"Results saved to {output_path_mmtom}")

if __name__ == "__main__":
    main()