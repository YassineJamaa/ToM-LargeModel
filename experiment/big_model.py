import os
import pickle
import torch
from dotenv import load_dotenv
from transformers import (LlavaNextVideoProcessor, 
                          LlavaNextVideoForConditionalGeneration, 
                          AutoTokenizer, 
                          AutoProcessor, 
                          MllamaForConditionalGeneration, 
                          AutoModelForCausalLM,
                          AutoConfig,
                          AutoModelForCausalLM)
from huggingface_hub import snapshot_download
import sys
from typing import Optional
import argparse
import pandas as pd
import transformers
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LocImportantUnits,
    ImportModel,
    LayerUnits,
    ZeroingAblation,
    AssessMMToM,
    AverageTaskStimuli,
    ToMLocDataset,
    ExtendedTomLocGPT4,
    setup_environment,
    load_chat_template,
)
from benchmark import BenchmarkMMToMQA, BenchmarkToMi, BenchmarkFanToM, BenchmarkOpenToM
from hardware.device_manager import setup_device
from src.assess import call_assessment


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

def main():
    token, cache_dir = setup_environment()
    device, device_ids = setup_device() 
    chat_template = load_chat_template("dataset/save_checkpoints/vlm_chat_template.txt")
    checkpoint = "llava-hf/LLaVA-NeXT-Video-34B-DPO-hf"
    model_type = "VLM"
    config = AutoConfig.from_pretrained(checkpoint, cache_dir=cache_dir, token=token)
    config.tie_word_embeddings = False
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(checkpoint, config=config, torch_dtype="float16", device_map="auto", cache_dir=cache_dir, token=token)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir, token=token)
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir, token=token)
    print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    vlm = ImportModel(model_type, model, tokenizer, processor, chat_template)

    # Classic & Extended ToM
    class_tom = ToMLocDataset()
    ext_tom = ExtendedTomLocGPT4()

    # MMToMQA Benchmark
    mmtom = BenchmarkMMToMQA()

    # Classic & Extended ToM
    units = LayerUnits(vlm, ext_tom, device)

    loc = LocImportantUnits(checkpoint, units.data_activation)
    ablate = ZeroingAblation()
    init_args = {
        "import_model": vlm,
        "mmtom": mmtom,
        "loc_units": loc,
        "ablation": ablate,
        "device":device,
        "is_video": True
    }
    method_args = {
        "modality": "text"
    }
    res = call_assessment(mmtom, "experiment", init_args, method_args)
    res.to_csv("MMToMQA_Llava-next-video-34B_zeroing_text.csv", index=False)

if __name__ == "__main__":
    main()




    