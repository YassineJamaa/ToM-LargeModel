from benchmark import BenchmarkText, BenchmarkVisionText, BenchmarkBaseline, BenchmarkMMToMQA
from src.hf_model import ImportModel
from src.localisation import LocImportantUnits
from src.ablation import ZeroingAblation
import pandas as pd
from collections import OrderedDict
from itertools import islice
import numpy as np
from tqdm import tqdm
from typing import Optional, Set, List
import torch
from transformers import TopKLogitsWarper

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
                                                                num_random=10)
        
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
    


class Assessmentv2:
    def __init__(self,
                 import_model: ImportModel,
                 # loc_units: LocImportantUnits,
                 ablation: ZeroingAblation,
                 device: str,
                 ):
        self.import_model = import_model
        # self.loc_units = loc_units
        self.mask_top = None
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
    def set_mask_top(self, mask):
        self.mask_top = mask
    
    def get_random_mask(self, seed=42):
        # Set the seed of the reproducibility
        if seed:
            np.random.seed(seed)

        # Calculate the total number of units
        percentage = np.sum(self.mask_top) / self.mask_top.size
        num_units_to_select = int(self.mask_top.size * percentage)

        # Create a flattened array of zeros
        mask_flat = np.zeros(self.mask_top.size, dtype=int)

        # Randomly select indices and set them to 1
        selected_indices = np.random.choice(self.mask_top.size, num_units_to_select, replace=False)
        mask_flat[selected_indices] = 1

        # Reshape the mask back to the original shape
        return mask_flat.reshape(self.mask_top.shape)

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
    def generate_ablation_masks(self, num_random=6, base_seed=42):
        seeds = [base_seed + i * 1000 for i in range(num_random)]
        percentage = self.mask_top.sum()/self.mask_top.size
        assess_dict = OrderedDict([
            ("no_ablation", None),
            (f"ablate_top", self.mask_top.T),
        ])
        assess_info = [
            "No Ablation",
            f"Ablation Top-{percentage:.2%}%",
        ]
        
        for idx, seed in enumerate(seeds, start=1):
            assess_dict[f"ablate_random_{idx}"] = self.get_random_mask(seed=seed).T
            assess_info.append(f"Random Ablation {percentage:.2%}% ({idx}/{num_random})")
        
        return assess_dict, assess_info
    
    def experiment(self,
                   benchmark: BenchmarkBaseline,
                   mask_top: np.ndarray,
                   modality: str="vision-text",
                   num_random: int=2,
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
        self.set_mask_top(mask_top)
        assess_dict, assess_info = self.generate_ablation_masks(num_random=num_random)
        
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

class Assessmentv3:
    def __init__(self,
                 import_model: ImportModel,
                 ablation: ZeroingAblation,
                 device: str):
        self.import_model = import_model
        self.mask_top = None
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
        if isinstance(self.benchmark, BenchmarkText) or self.import_model.model_type=="LLM":
            self.modality = "text_only"
        elif isinstance(self.benchmark, BenchmarkVisionText):
            if modality in {"text-only", "vision-text"}: # Allow the user to specify "text-only" or "vision-text" for BenchmarkVisionText
                self.modality = modality
            else:
                raise ValueError("For BenchmarkVisionText, modality must be 'text-only' or 'vision-text'.")
        
    def set_mask_top(self, mask):
        self.mask_top = mask
    
    def get_random_mask(self, seed=42):
        if seed: # Set the seed of the reproducibility
            np.random.seed(seed)
        percentage = np.sum(self.mask_top) / self.mask_top.size # Calculate the total number of units
        num_units_to_select = int(self.mask_top.size * percentage)
        mask_flat = np.zeros(self.mask_top.size, dtype=int) # Create a flattened array of zeros
        selected_indices = np.random.choice(self.mask_top.size, num_units_to_select, replace=False) # Randomly select indices and set them to 1
        mask_flat[selected_indices] = 1
        return mask_flat.reshape(self.mask_top.shape) # Reshape the mask back to the original shape

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
        content = [{"type": "text", "text": text}]
        processor_args = {}
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
        if self.modality == "vision-text" and "frames" in row:
            inputs_args["frames"] = row["frames"]
        inputs = self.get_inputs(**inputs_args)

        with torch.no_grad(): # Inference
            outputs = self.import_model.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=1, 
                    temperature = None,
                    top_p = None,
                    do_sample=False,
                    top_k = None,
                    pad_token_id=self.import_model.tokenizer.eos_token_id
                )
            decoded_token = self.import_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_token[-1]
    
    def inference(self, mask: np.ndarray=None):
        """ Collect the Log Softmax of the mask """
        predictions = []
        self.ablation.clear_hooks(self.import_model)
        if mask is not None: # I. Set the mask if present
            language_model = self.import_model.get_language_model()
            for idx, layer in enumerate(language_model.model.layers):
                layer.register_forward_hook(self.ablation.get_hook_ablate(idx, mask))
        for idx in tqdm(range(len(self.data))): # II. Mask'inference
            prediction = self.inference_row(idx)
            predictions.append(prediction)
        return predictions

    def generate_ablation_masks(self, num_random=6, no_lesion=True, add_lesion=True, base_seed=42):
        seeds = [base_seed + i * 1000 for i in range(num_random)]
        percentage = self.mask_top.sum()/self.mask_top.size
        assess_dict = OrderedDict([])
        assess_info = []
        if no_lesion:
            assess_dict["no_ablation"] = None
            assess_info.append("No Ablation")
        if add_lesion:
            assess_dict[f"ablate_top"] = self.mask_top.T
            assess_info.append(f"Ablation Top-{percentage:.2%}%")
            for idx, seed in enumerate(seeds, start=1):
                assess_dict[f"ablate_random_{idx}"] = self.get_random_mask(seed=seed).T
                assess_info.append(f"Random Ablation {percentage:.2%}% ({idx}/{num_random})")
        return assess_dict, assess_info

    def experiment(self,
                   benchmark: BenchmarkBaseline,
                   mask_top: np.ndarray,
                   modality: str="vision-text",
                   num_random: int=2,
                   add_lesion: bool=True,
                   no_lesion: bool= True,
                   inference_mode: str="normal",
                   ):
        """
        Conduct an experiment based on the given benchmark.

        Args:
            benchmark (BenchmarkBaseline): The benchmark object, either BenchmarkText or BenchmarkVisionText.
            modality (str, optional): Modality of the experiment. Must be "text-only" or "vision-text" for 
                                    BenchmarkVisionText, and is set to "text-only" for BenchmarkText.
            percentage (float, optional): Fix the top-% neural units that we are going to ablate.
        """
        self.benchmark = benchmark # 0. Set Benchmark
        self.set_modality(modality) # I. set the modality
        self.ablation.clear_hooks(self.import_model) # Clear model's hooks
        self.import_model.model.eval() # Set the model to evaluation mode
        self.set_mask_top(mask_top) # Set mask
        assess_dict, assess_info = self.generate_ablation_masks(num_random=num_random, no_lesion=no_lesion, add_lesion=add_lesion)
        self.data = self.benchmark.data.copy() # Set Dataset

        # V. Inference
        if inference_mode == "normal":
            for idx, (key, mask) in islice(enumerate(assess_dict.items()), 0, None):
                print(f"Step {idx+1}/{len(assess_info)}: {assess_info[idx]}")
                predictions = self.inference(mask)
                self.data[f"predict_{key}"] = predictions
            return self.data
        elif inference_mode == "quick":
            pass
            # Write the quick version


class Assessmentv4:
    def __init__(self,
                 import_model: ImportModel,
                 ablation: ZeroingAblation,
                 device: str):
        self.import_model = import_model
        self.mask_top = None
        self.ablation = ablation
        self.device = device
        self.modality = None
        self.data = None
        self.benchmark = None
        self.generated_tokens = 1
        if self.import_model.model_type=="LLM":
            self.get_inputs = self.get_inputs_text
        elif self.import_model.model_type=="VLM":
            self.get_inputs = self.get_inputs_vision_text
        else:
            self.get_inputs = None
        
    def set_generated_tokens(self, generated_tokens: int):
        self.generated_tokens = generated_tokens
    
    def set_modality(self, modality):
        if isinstance(self.benchmark, BenchmarkText) or self.import_model.model_type=="LLM":
            self.modality = "text_only"
        elif isinstance(self.benchmark, BenchmarkVisionText):
            if modality in {"text-only", "vision-text"}: # Allow the user to specify "text-only" or "vision-text" for BenchmarkVisionText
                self.modality = modality
            else:
                raise ValueError("For BenchmarkVisionText, modality must be 'text-only' or 'vision-text'.")
        
    def set_mask_top(self, mask):
        self.mask_top = mask
    
    def get_random_mask(self, seed=42):
        if seed: # Set the seed of the reproducibility
            np.random.seed(seed)
        percentage = np.sum(self.mask_top) / self.mask_top.size # Calculate the total number of units
        num_units_to_select = int(self.mask_top.size * percentage)
        mask_flat = np.zeros(self.mask_top.size, dtype=int) # Create a flattened array of zeros
        selected_indices = np.random.choice(self.mask_top.size, num_units_to_select, replace=False) # Randomly select indices and set them to 1
        mask_flat[selected_indices] = 1
        return mask_flat.reshape(self.mask_top.shape) # Reshape the mask back to the original shape

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
        args_processor = {"text": text}
        if frames:
            args_processor["images"] = frames
        inputs = self.import_model.processor(**args_processor, return_tensors="pt").to(self.device)
        return inputs
    
    def inference_row(self, position: int):
        """
        Process a row from the benchmark and call the return_inputs function with dynamic arguments.
        """
        row = self.benchmark[position]
        inputs_args = {"text": row["text"]}
        if self.modality == "vision-text" and "frames" in row:
            inputs_args["frames"] = row["frames"]
        inputs = self.get_inputs(**inputs_args)

        with torch.no_grad(): # Inference
            outputs = self.import_model.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.generated_tokens, 
                    temperature = None,
                    top_p = None,
                    do_sample=False,
                    top_k = None,
                    pad_token_id=self.import_model.tokenizer.eos_token_id
                )
            decoded_token = self.import_model.tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
        return decoded_token[-1].strip()
    
    def inference(self, mask: np.ndarray=None):
        """ Collect the Log Softmax of the mask """
        predictions = []
        self.ablation.clear_hooks(self.import_model)
        if mask is not None: # I. Set the mask if present
            language_model = self.import_model.get_language_model()
            for idx, layer in enumerate(language_model.model.layers):
                layer.register_forward_hook(self.ablation.get_hook_ablate(idx, mask))
        for idx in tqdm(range(len(self.data))): # II. Mask'inference
            prediction = self.inference_row(idx)
            predictions.append(prediction)
        return predictions

    def generate_ablation_masks(self, num_random=6, no_lesion=True, add_lesion=True, base_seed=42):
        seeds = [base_seed + i * 1000 for i in range(num_random)]
        percentage = self.mask_top.sum()/self.mask_top.size
        assess_dict = OrderedDict([])
        assess_info = []
        if no_lesion:
            assess_dict["no_ablation"] = None
            assess_info.append("No Ablation")
        if add_lesion:
            assess_dict[f"ablate_top"] = self.mask_top.T
            assess_info.append(f"Ablation Top-{percentage:.2%}%")
            for idx, seed in enumerate(seeds, start=1):
                assess_dict[f"ablate_random_{idx}"] = self.get_random_mask(seed=seed).T
                assess_info.append(f"Random Ablation {percentage:.2%}% ({idx}/{num_random})")
        return assess_dict, assess_info

    def experiment(self,
                   benchmark: BenchmarkBaseline,
                   mask_top: np.ndarray,
                   modality: str="vision-text",
                   num_random: int=2,
                   add_lesion: bool=True,
                   no_lesion: bool= True,
                   inference_mode: str="normal",
                   ):
        """
        Conduct an experiment based on the given benchmark.

        Args:
            benchmark (BenchmarkBaseline): The benchmark object, either BenchmarkText or BenchmarkVisionText.
            modality (str, optional): Modality of the experiment. Must be "text-only" or "vision-text" for 
                                    BenchmarkVisionText, and is set to "text-only" for BenchmarkText.
            percentage (float, optional): Fix the top-% neural units that we are going to ablate.
        """
        self.benchmark = benchmark # 0. Set Benchmark
        self.set_modality(modality) # I. set the modality
        self.ablation.clear_hooks(self.import_model) # Clear model's hooks
        self.import_model.model.eval() # Set the model to evaluation mode
        self.set_mask_top(mask_top) # Set mask
        assess_dict, assess_info = self.generate_ablation_masks(num_random=num_random, no_lesion=no_lesion, add_lesion=add_lesion)
        self.data = self.benchmark.data.copy() # Set Dataset

        # V. Inference
        if inference_mode == "normal":
            for idx, (key, mask) in islice(enumerate(assess_dict.items()), 0, None):
                print(f"Step {idx+1}/{len(assess_info)}: {assess_info[idx]}")
                predictions = self.inference(mask)
                self.data[f"predict_{key}"] = predictions
            return self.data
        elif inference_mode == "quick":
            pass


class AssessmentTopK:
    def __init__(self,
                 import_model: ImportModel,
                 ablation: ZeroingAblation,
                 target_chars: Set[str],
                 device: str):
        self.import_model = import_model
        self.mask_top = None
        self.ablation = ablation
        self.device = device
        self.modality = None
        self.data = None
        self.benchmark = None
        self.target_chars = target_chars
        self.topK = None
        self.generated_tokens = 1
        if self.import_model.model_type=="LLM":
            self.get_inputs = self.get_inputs_text
        elif self.import_model.model_type=="VLM":
            self.get_inputs = self.get_inputs_vision_text
        else:
            self.get_inputs = None
        
    def set_generated_tokens(self, generated_tokens: int):
        self.generated_tokens = generated_tokens
    
    def set_modality(self, modality):
        if isinstance(self.benchmark, BenchmarkText) or self.import_model.model_type=="LLM":
            self.modality = "text_only"
        elif isinstance(self.benchmark, BenchmarkVisionText):
            if modality in {"text-only", "vision-text"}: # Allow the user to specify "text-only" or "vision-text" for BenchmarkVisionText
                self.modality = modality
            else:
                raise ValueError("For BenchmarkVisionText, modality must be 'text-only' or 'vision-text'.")
        
    def set_mask_top(self, mask):
        self.mask_top = mask
    
    def get_random_mask(self, seed=42):
        if seed: # Set the seed of the reproducibility
            np.random.seed(seed)
        percentage = np.sum(self.mask_top) / self.mask_top.size # Calculate the total number of units
        num_units_to_select = int(self.mask_top.size * percentage)
        mask_flat = np.zeros(self.mask_top.size, dtype=int) # Create a flattened array of zeros
        selected_indices = np.random.choice(self.mask_top.size, num_units_to_select, replace=False) # Randomly select indices and set them to 1
        mask_flat[selected_indices] = 1
        return mask_flat.reshape(self.mask_top.shape) # Reshape the mask back to the original shape

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
        args_processor = {"text": text}
        if frames:
            args_processor["images"] = frames
        inputs = self.import_model.processor(**args_processor, return_tensors="pt").to(self.device)
        return inputs
    
    def inference_row(self, position: int):
        """
        Process a row from the benchmark and call the return_inputs function with dynamic arguments.
        """
        row = self.benchmark[position]
        inputs_args = {"text": row["text"]}
        if self.modality == "vision-text" and "frames" in row:
            inputs_args["frames"] = row["frames"]
        inputs = self.get_inputs(**inputs_args)

        logits_processor = [TopKLogitsWarper(top_k=self.topK)]
        with torch.no_grad(): # Inference
            outputs = self.import_model.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.generated_tokens, 
                    temperature = None,
                    top_p = None,
                    do_sample=False,
                    top_k = None,
                    output_scores=True, 
                    pad_token_id=self.import_model.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    logits_processor=logits_processor
                )
            logits = outputs.scores[0] # Compute the logits
            probs = torch.softmax(logits, dim=-1) # Compute the probabilities

            top_indices = torch.nonzero(probs, as_tuple=True)[1] # Take only the top indices
            top_indices = top_indices[torch.argsort(probs[0, top_indices], descending=True)] # Sort them
            top_probs = probs[0, top_indices].tolist() # take the probs
        return (top_indices, top_probs)
    
    def inference(self, mask: np.ndarray=None):
        """ Collect the Log Softmax of the mask """
        top_tokens = []
        self.ablation.clear_hooks(self.import_model)
        if mask is not None: # I. Set the mask if present
            language_model = self.import_model.get_language_model()
            for idx, layer in enumerate(language_model.model.layers):
                layer.register_forward_hook(self.ablation.get_hook_ablate(idx, mask))
        for idx in tqdm(range(len(self.data))): # II. Mask'inference
            top_tokens.append(self.inference_row(idx))
        return top_tokens

    def generate_ablation_masks(self, num_random=6, no_lesion=True, add_lesion=True, base_seed=42):
        seeds = [base_seed + i * 1000 for i in range(num_random)]
        percentage = self.mask_top.sum()/self.mask_top.size
        assess_dict = OrderedDict([])
        assess_info = []
        if no_lesion:
            assess_dict["no_ablation"] = None
            assess_info.append("No Ablation")
        if add_lesion:
            assess_dict[f"ablate_top"] = self.mask_top.T
            assess_info.append(f"Ablation Top-{percentage:.2%}%")
            for idx, seed in enumerate(seeds, start=1):
                assess_dict[f"ablate_random_{idx}"] = self.get_random_mask(seed=seed).T
                assess_info.append(f"Random Ablation {percentage:.2%}% ({idx}/{num_random})")
        return assess_dict, assess_info

    def find_best_candidate(self, top_tokens):
        def better_candidates(tokens: list):
            # Find the first occurrence of the target characters
            selected_token = tokens[0]
            for token in tokens:
                # Check if the token matches exactly "a", "b", "c", or "d"
                if token.strip() in self.target_chars:
                    selected_token = token.strip()  # Strip spaces if needed
                    break
            return selected_token
        predictions = []
        for indices, probs in top_tokens:
            tokens = [self.import_model.tokenizer.decode([token_id]).strip().lower() for token_id in indices]
            predictions.append(better_candidates(tokens))
        return predictions

    def experiment(self,
                   benchmark: BenchmarkBaseline,
                   mask_top: np.ndarray,
                   modality: str="vision-text",
                   num_random: int=2,
                   add_lesion: bool=True,
                   no_lesion: bool= True,
                   inference_mode: str="normal",
                   topK=5,
                   ):
        """
        Conduct an experiment based on the given benchmark.

        Args:
            benchmark (BenchmarkBaseline): The benchmark object, either BenchmarkText or BenchmarkVisionText.
            modality (str, optional): Modality of the experiment. Must be "text-only" or "vision-text" for 
                                    BenchmarkVisionText, and is set to "text-only" for BenchmarkText.
            percentage (float, optional): Fix the top-% neural units that we are going to ablate.
        """
        self.topK = topK
        self.benchmark = benchmark # 0. Set Benchmark
        self.set_modality(modality) # I. set the modality
        self.ablation.clear_hooks(self.import_model) # Clear model's hooks
        self.import_model.model.eval() # Set the model to evaluation mode
        self.set_mask_top(mask_top) # Set mask
        assess_dict, assess_info = self.generate_ablation_masks(num_random=num_random, no_lesion=no_lesion, add_lesion=add_lesion)
        self.data = self.benchmark.data.copy() # Set Dataset

        # V. Inference
        if inference_mode == "normal":
            for idx, (key, mask) in islice(enumerate(assess_dict.items()), 0, None):
                print(f"Step {idx+1}/{len(assess_info)}: {assess_info[idx]}")
                top_tokens = self.inference(mask)
                self.data[f"predict_{key}"] = self.find_best_candidate(top_tokens)
            return self.data
        elif inference_mode == "quick":
            pass