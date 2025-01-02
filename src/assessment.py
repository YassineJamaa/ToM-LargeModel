from benchmark import BenchmarkText, BenchmarkVisionText, BenchmarkBaseline, BenchmarkMMToMQA
from src.hf_model import ImportModel
from src.localisation import LocImportantUnits
from src.ablation import ZeroingAblation
import pandas as pd
from collections import OrderedDict
from itertools import islice
import numpy as np
from tqdm import tqdm
from typing import Optional
import torch

def get_highest_key(probabilities):
    """
    This function takes a dictionary of probabilities as input and returns the key with the highest value.
    
    Steps:
    1. Identify the key corresponding to the maximum value in the dictionary using the `max` function.
    2. Normalize the key by stripping any leading/trailing whitespace and converting it to lowercase.
    
    Args:
        probabilities (dict): A dictionary where keys are strings and values are numeric probabilities.

    Returns:
        str: The key with the highest probability value, normalized (lowercase and stripped of whitespace).
    """
    max_key = max(probabilities, key=probabilities.get)
    return max_key.strip().lower()

class Assessment:
    def __init__(self,
                 import_model: ImportModel,
                 loc_units: LocImportantUnits,
                 ablation: ZeroingAblation,
                 device: str,
                 ):
        self.import_model = import_model
        self.loc_units = loc_units
        self.ablation = ablation
        self.device = device
        self.modality = None
        self.data = None
        self.benchmark = None
        if self.import_model.model_type=="LLM":
            self.get_inputs = self.get_inputs_text
        elif self.import_model.model_type=="VLM":
            self.get_inputs = self.get_inputs_vision_text
        else:
            self.get_inputs = None
    
    def set_modality(self, modality):
        if modality in ["text-only", "vision-text"]:
            self.modality=modality
        else:
            raise ValueError(
                f"Modality '{modality}' not recognized. Please choose either 'text-only' or 'vision-text'."
            )
    
    def set_tokens_ids(self, benchmark_df: pd.DataFrame):
        def generate_token_ids(cands_letter):
            tokens_list = [self.import_model.tokenizer.tokenize(f" {s}") for s in cands_letter]
            token_ids_list = [self.import_model.tokenizer.convert_tokens_to_ids(tokens) 
                                        for tokens in tokens_list]
            return token_ids_list
        
        benchmark_df["token_ids"] = benchmark_df["cands_letter"].apply(generate_token_ids)
        return benchmark_df
    
    def get_inputs_text(self, text):
        """
        Prepares model inputs based on the user's text (Text-only).

        Args:
            text (str): User's text input.
            frames (list, optional): List of image or video frames.

        Returns:
            dict: Processed inputs for the model.
        """
        inputs = self.import_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return inputs

    def get_inputs_vision_text(self, text, frames: Optional[list] = None):
        """
        Prepares model inputs based on the user's text and optional media (images or videos).

        Args:
            text (str): User's text input.
            frames (list, optional): List of image or video frames.

        Returns:
            dict: Processed inputs for the model.
        """

        # Handle text input with optional single or multiple images
        content = [{"type": "text", "text": text}]
        processor_args = {}
        # if frames and not self.is_video:
        #     content.extend([{"type": "image"} for _ in frames])
        if frames:
            content.extend([{"type": "video"}])
            clip = np.stack([frame for frame in frames])
            processor_args["videos"] = clip

        conversation = [{"role": "user", "content": content}]
        prompt = self.import_model.method_process.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.import_model.method_process(
            text=prompt,
            **processor_args,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs   

    def inference_row(self, position: int):
        """
        Process a row from the benchmark and call the return_inputs function with dynamic arguments.
        """
        row = self.benchmark[position]
        inputs_args = {"text": row["text"]}

        # Dynamically prepare arguments
        if self.modality == "vision-text" and "frames" in row:
            inputs_args["frames"] = row["frames"]

        inputs = self.get_inputs(**inputs_args)
        with torch.no_grad():
            outputs = self.import_model.model.generate(
                                        input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        do_sample=False,
                                        temperature=None,  # Unset temperature
                                        top_p=None,
                                        max_new_tokens=1,  # Generate exactly one additional token for each input
                                        output_scores=True,
                                        pad_token_id=self.import_model.tokenizer.pad_token_id,
                                        return_dict_in_generate=True)
            logits = outputs.scores[0]
            prob_tokens = torch.log_softmax(logits, dim=1)[0]
        
        cands_probs = {token: prob_tokens[token_id[0]].item() for token, token_id in zip(self.data["cands_letter"].iloc[position], self.data["token_ids"].iloc[position])}
        return cands_probs
    
    def inference(self, mask: np.ndarray=None):
        """ Collect the Log Softmax of the mask """
        result_mask = []
        self.ablation.clear_hooks(self.import_model)
        # I. Set the mask if present
        if mask is not None:
            language_model = self.import_model.get_language_model()
            for idx, layer in enumerate(language_model.model.layers):
                layer.register_forward_hook(self.ablation.get_hook_ablate(idx, mask))

        # II. Mask'inference
        for idx in tqdm(range(len(self.data))):
            probs = self.inference_row(idx)
            result_mask.append(probs)
        return result_mask
    
    # Define a function to generate random seeds
    def generate_ablation_masks(self, percentage, num_random=6, base_seed=42):
        seeds = [base_seed + i * 1000 for i in range(num_random)]
        assess_dict = OrderedDict([
            ("no_ablation", None),
            (f"ablate_top", self.loc_units.get_masked_ktop(percentage).T),
        ])
        assess_info = [
            "No Ablation",
            f"Ablation Top-{percentage:.2%}%",
        ]
        
        for idx, seed in enumerate(seeds, start=1):
            assess_dict[f"ablate_random_{idx}"] = self.loc_units.get_random_mask(percentage, seed=seed).T
            assess_info.append(f"Random Ablation {percentage:.2%}% ({idx}/{num_random})")
        
        return assess_dict, assess_info
    
    def experiment(self,
                   benchmark: BenchmarkBaseline,
                   modality: str="vision-text",
                   percentage: float=0.01,
                   ):
        """
        Conduct an experiment based on the given benchmark.

        Args:
            benchmark (BenchmarkBaseline): The benchmark object, either BenchmarkText or BenchmarkVisionText.
            modality (str, optional): Modality of the experiment. Must be "text-only" or "vision-text" for 
                                    BenchmarkVisionText, and is set to "text-only" for BenchmarkText.
            percentage (float, optional): Fix the top-% neural units that we are going to ablate.
        """
        # 0. Benchmark: Remove "Answer :"?
        self.benchmark = benchmark
        
        # I. set the modality
        if isinstance(benchmark, BenchmarkText) or self.import_model.model_type=="LLM":
            # Force modality to "text-only" for BenchmarkText
            self.set_modality("text-only")
        elif isinstance(benchmark, BenchmarkVisionText):
            # Allow the user to specify "text-only" or "vision-text" for BenchmarkVisionText
            if modality in {"text-only", "vision-text"}:
                self.set_modality(modality)
            else:
                raise ValueError("For BenchmarkVisionText, modality must be 'text-only' or 'vision-text'.")

        # II. Clear model's hooks
        self.ablation.clear_hooks(self.import_model)
        self.import_model.model.eval()

        # III. Set assessment script: No ablation, top-k% ablation, random-k% ablation 1, ...
        assess_dict, assess_info = self.generate_ablation_masks(percentage=percentage,
                                                                num_random=3)
        
        # IV. Process benchmark data: Get the token ids for each candidates
        data = self.benchmark.data.copy()
        self.data = self.set_tokens_ids(data).copy()

        # V. Inference
        for idx, (key, mask) in islice(enumerate(assess_dict.items()), 0, None):
            print(f"Step {idx+1}/{len(assess_info)}: {assess_info[idx]}")
            result_mask = self.inference(mask)
            data[f"prob_tokens_{key}"] = result_mask
            data[f"predict_{key}"] = data[f"prob_tokens_{key}"].apply(get_highest_key)
        return data