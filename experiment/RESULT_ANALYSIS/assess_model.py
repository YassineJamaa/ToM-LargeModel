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
import json

from benchmark import BenchmarkMMToMQA, BenchmarkToMi, BenchmarkOpenToM, BenchmarkFanToM, BenchmarkBaseline, BenchmarkVisionText, ToMiPromptEngineering, MMToMQAPromptEngineering, MATHBenchmark
from src import (
                AssessmentTopK,
                Assessmentv4,
                setup_environment,
                 ImportModel,
                 load_chat_template,
                 LayerUnits,
                 LocImportantUnits,
                 ZeroingAblation,
                 ToMLocDataset,
                 AverageTaskStimuli,
                 MeanImputation,
                 ExtendedTomLocGPT4,
                 MDLocDataset,
                 get_masked_kbot,
                 get_masked_ktop)
from analysis import set_model

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run experiments with language models.")
    parser.add_argument("--model_func", type=str, default="AutoModelForCausalLM", help="Mode processing function for VLM")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model checkpoint path.")
    parser.add_argument("--cache", type=str, default="CACHE_DIR", help="Temporary: two cache.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_checkpoint = args.model_name
    model_func = args.model_func
    model_type = "LLM"
    cache = args.cache

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

    # ToM Localizer
    classic_tom = ToMLocDataset()
    classic_units = LayerUnits(import_model = llm, localizer = classic_tom, device = device)
    classic_loc = LocImportantUnits("classic", classic_units.data_activation)

    # MD Localizer
    md = MDLocDataset()
    md_units = LayerUnits(import_model = llm, localizer = md, device = device)
    md_loc = LocImportantUnits("classic", md_units.data_activation)

    