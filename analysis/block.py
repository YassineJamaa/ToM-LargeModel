import os
import torch
from dotenv import load_dotenv
import sys
import argparse
import numpy as np
import importlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoProcessor)

from benchmark import BenchmarkBaseline

from src import (ImportModel,
                 LayerUnits,
                 AverageTaskStimuli,
                 LocImportantUnits,
                 ToMLocDataset,
                 ExtendedTomLocGPT4,
                 ZeroingAblation,
                 MeanImputation,
                 Assessment,
                 load_chat_template)

def set_model(model_checkpoint: str,
              model_type: str,
              token_hf: str,
              cache_dir: str,
              model_func: str="AutoModelForCausalLM"):
    """ Set the class ImportModel. """
    # dynamically use the model loader
    module = importlib.import_module("transformers")
    model_fn = getattr(module, model_func)
    
    # Set the config
    config = AutoConfig.from_pretrained(model_checkpoint, cache_dir=cache_dir, token=token_hf)
    config.tie_word_embeddings = False
    # config.torch_dtype = "bfloat16"
    
    if model_type=="LLM":
        model = model_fn.from_pretrained(model_checkpoint,
                                         torch_dtype="bfloat16",
                                        device_map="auto",
                                        cache_dir=cache_dir,
                                        token=token_hf)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                  cache_dir=cache_dir,
                                                  token=token_hf)
        import_model = ImportModel(model_type=model_type,
                                    model=model,
                                    tokenizer=tokenizer)
    elif model_type == "VLM":
        chat_template = load_chat_template("dataset/save_checkpoints/vlm_chat_template.txt")

        model = model_fn.from_pretrained(model_checkpoint,
                                        device_map="auto",
                                        cache_dir=cache_dir,
                                        token=token_hf)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, 
                                                  cache_dir=cache_dir, 
                                                  token=token_hf)
        
        processor = AutoProcessor.from_pretrained(model_checkpoint, 
                                                cache_dir=cache_dir, 
                                                token=token_hf)
        import_model = ImportModel(model_type,
                                   model,
                                   tokenizer,
                                   processor,
                                   chat_template)
    else:
        raise ValueError(f"{model_type} is not a model architecture supported.")

    return import_model


class AnalysisBlock:
    def __init__(self, 
                 model_args: dict,
                 device: str,
                 ):
        self.import_model = set_model(**model_args)
        self.device = device
        self.modality = None

        # Set the Classic and Extended Localizer
        self.class_tom = ToMLocDataset()
        self.ext_tom = ExtendedTomLocGPT4()

        # Units for Classic & Extended Localizers
        self.classic_units = LayerUnits(self.import_model,
                                        self.class_tom,
                                        self.device)
        self.extended_units = LayerUnits(self.import_model,
                                        self.ext_tom,
                                        self.device)
        
        # Imputation: Zeroing & Mean Imputation
        self.zeroing = ZeroingAblation()
        self.mean_imputation = None
    
    def set_mean_imputation(self, 
                            benchmark: BenchmarkBaseline):
        avg_stim = AverageTaskStimuli(benchmark, 
                                      self.import_model, 
                                      self.device)
        mean_units = avg_stim.extract_all_units()
        self.mean_imputation = MeanImputation(mean_units=mean_units)
    
    def get_ablation(self, imputation="zeroing"):
        if imputation=="zeroing":
            return self.zeroing
        elif imputation=="mean":
            return self.mean_imputation
        else:
            raise ValueError(f"{imputation} is not an imputation supported.")
        
    
    def get_units(self, kind: str="classic"):
        if kind == "classic":
            data_activation = self.classic_units.data_activation
        elif kind == "extended":
            data_activation = self.extended_units.data_activation
        return data_activation

   
    def get_layer_units(self,
                        percentage: float,
                        kind: str="classic"):
        data_activation = self.get_units(kind)
        loc = LocImportantUnits("model", data_activation)
        mask = loc.get_masked_ktop(percentage)

        # Calculate the percentage of important units for each layer
        layer_percentage = [(np.sum(layer) / layer.size) * 100 for layer in mask.T]
        layer_percentage = np.array(layer_percentage).reshape(-1, 1)
        layer_dict = {f"layer_{i}": layer_percentage[i, 0] for i in range(layer_percentage.shape[0])}
        return layer_dict
    
    def run(self,
            benchmark: BenchmarkBaseline,
            imputation: str="zeroing",
            percentage: float=0.01,
            kind: str="classic",
            modality: str="vision-text",
            ):
        data_activation = self.get_units(kind)
        loc = LocImportantUnits("model", data_activation)
        assess = Assessment(import_model=self.import_model,
                            loc_units=loc,
                            ablation=self.get_ablation(imputation),
                            device=self.device)
        result = assess.experiment(percentage=percentage,
                                benchmark=benchmark,
                                modality=modality)
        return result
        


        






        




