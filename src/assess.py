from src.localisation import LocImportantUnits
from src.huggingface_models import ImportLLM, ImportVLM
from benchmark import BenchmarkBaseline, BenchmarkMMToMQA
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from typing import Optional
from collections import OrderedDict
import os
from itertools import islice
from src.ablation import ZeroingAblation

def compute_cand_score(row):
    logits = row['log_sm']
    n = row['last_tokens']
    last_token = row["last_token_position"]
    if n == 1:
        return logits[last_token-1] # Return the last element
    elif n > 1:
        return sum(logits[last_token-n:last_token]) / n  # Return the mean of the last N elements

class AssessBenchmark:
    def __init__(self,
                 llm: ImportLLM,
                 loc_units: LocImportantUnits,
                 ablation: ZeroingAblation):
        self.llm = llm
        self.loc_units = loc_units
        self.ablation = ablation

    def assess(self,
           data: pd.DataFrame,
           mask: Optional[np.ndarray] = None,
           batch_size: int = 20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.model.to(device)
        self.ablation.clear_hooks(self.llm)

        if mask is not None:
            for idx, layer in enumerate(self.llm.model.model.layers):
                layer.register_forward_hook(self.ablation.get_hook_ablate(idx, mask))

        exp_df = data.copy()
        logsm_list = []
        last_token_position_list = []

        for i in tqdm(range(0, len(exp_df), batch_size)):
            batch_texts = exp_df["fusion"].tolist()[i:i+batch_size]
            # Tokenize the batch of texts and get input IDs
            inputs = self.llm.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True) # remove truncation=True
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU

            with torch.no_grad():
                outputs = self.llm.model(**inputs)
                logprobs = torch.log_softmax(outputs.logits, dim=-1).cpu()  # Move logprobs to CPU
                logprobs_ids = torch.gather(logprobs[:,:-1], 2, inputs["input_ids"][:,1:].cpu().unsqueeze(-1)).squeeze(-1)

            last_token_position = (inputs["attention_mask"].sum(dim=1) - 1).cpu()  # Move to CPU
            last_token_position_list.extend(last_token_position.tolist())
            logsm_list.extend(logprobs_ids)

            del inputs, outputs, logprobs, logprobs_ids  # Free up GPU memory
            torch.cuda.empty_cache()  # Clear unused cached memory

        exp_df["log_sm"] = logsm_list
        exp_df["last_token_position"] = last_token_position_list
        exp_df["score_cand"] = exp_df.apply(compute_cand_score, axis=1)

        # Group by 'id_prompt' and find the row with the minimum surprisal for each group
        best_cand_df = exp_df.loc[exp_df.groupby('id_prompt')['score_cand'].idxmax()]
        best_cand_df = best_cand_df.reset_index(drop=True)
        subset = best_cand_df[["id_prompt", "cand"]]
        return subset

    def experiment(self,
                   bn_data: BenchmarkBaseline,
                   check_path: Optional[str]=None,
                   batch_size: int = 20,
                   pct=0.01):
        self.llm.model.eval()
        assess_dict = OrderedDict([
            ("no_ablation", None),
            (f"ablate_top", self.loc_units.get_masked_ktop(pct).T),
            (f"ablate_random1", self.loc_units.get_random_mask(pct, seed=42).T),
            (f"ablate_random2", self.loc_units.get_random_mask(pct, seed=12345).T),
            (f"ablate_random3", self.loc_units.get_random_mask(pct, seed=98765).T)
        ])
        info_process = ["No Ablation",
                        f"Ablation Top-{pct*100:.2f}%",
                        f"Random Ablation {pct*100:.2f}% (1/3)",
                        f"Random Ablation {pct*100:.2f}% (2/3)",
                        f"Random Ablation {pct*100:.2f}% (3/3)"]

        if check_path and os.path.exists(check_path):
            df = pd.read_csv(check_path)
            step = len(assess_dict)
            for idx, (key, val) in enumerate(assess_dict.items()):
                var_col = f"predict_{key}"
                if var_col not in df.columns:
                    step = idx
                    break
        else:
            df = bn_data.data.copy()
            step = 0
        exp_df = bn_data.expanded_df.copy()

        # Tokenize the "fusion" variable starting from "prompt_end_pos"
        exp_df["last_tokens"] = exp_df.apply(
            lambda row: self.llm.tokenizer(row["fusion"][row["prompt_end_pos"]:], add_special_tokens=False)["input_ids"], axis=1
        ).apply(len)
        df = df.reset_index(drop=True)
        for idx, (key, mask) in islice(enumerate(assess_dict.items()), step, None):
            # Display progress
            print(f"Step {idx+1}/{len(info_process)}: {info_process[idx]}")
            subset = self.assess(exp_df, mask, batch_size)
            df = df.merge(subset, left_index=True, right_on="id_prompt", how="left")
            df = df.rename(columns={"cand": f"predict_{key}"})
            df = df.drop(columns=["id_prompt"])
            # Save intermediate results if save_path is provided
            if check_path:
                df.to_csv(check_path, index=False)
        return df

class AssessMMToM:
    def __init__(self,
                 vlm: ImportVLM,
                 mmtom: BenchmarkMMToMQA,
                 loc_units: LocImportantUnits,
                 ablation: ZeroingAblation,
                 is_video=True):
        self.vlm = vlm
        self.mmtom = mmtom
        self.loc_units = loc_units
        self.ablation = ablation
        self.is_video = is_video
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def return_inputs(self, text, frames: Optional[list] = None):
        """
        Prepares model inputs based on the user's text and optional media (images or videos).

        Args:
            text (str): User's text input.
            frames (list, optional): List of image or video frames.

        Returns:
            dict: Processed inputs for the model.
        """
        if (frames is None) or (not self.is_video) or (len(frames)==1):
            # Handle text input with optional single or multiple images
            content = [{"type": "text", "text": text}]
            if frames:
                content.extend([{"type": "image"} for _ in frames])

            conversation = [{"role": "user", "content": content}]
            prompt = self.vlm.processor.apply_chat_template(conversation, add_generation_prompt=True)

            inputs = self.vlm.processor(
                text=prompt,
                images=frames if frames else None,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        
        elif self.is_video and frames and len(frames) > 1:
            # Handle video input
            clip = np.stack([np.array(frame) for frame in frames[:24]])

            conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "video"},
                        ],
                    }
                ]
            prompt = self.vlm.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.vlm.processor(
                text=prompt,
                videos=clip,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

        return inputs
    
    def compute_row(self, pos: int, threshold: int=24):
        """ Collect the Log Softmax for a pair (Frames, Text)"""
        text, frames = self.mmtom[pos]
        inputs = self.return_inputs(text, frames)

        # Generate the output
        with torch.no_grad():
            output = self.vlm.model(**inputs)
            logprobs = torch.log_softmax(output.logits, dim=-1).cpu()  # Move logprobs to CPU
        
        # Get the log_softmax for the options "a", "A", "b", "B"
        next_token_logits = logprobs[0, -1, :]  # Logits for the last position
        add_tokens = [" a", " b", " A", " B"] # List of candidates
        token_logsm_list = [(add_token, 
                             torch.log_softmax(next_token_logits, dim=-1)[self.vlm.tokenizer.encode(add_token, add_special_tokens=False)].item())
                             for add_token in add_tokens]
        return token_logsm_list
    
    def compute(self, mask: Optional[np.ndarray] = None,):
        result_list = []

        if mask is not None:
            for idx, layer in enumerate(self.vlm.model.language_model.model.layers):
                layer.register_forward_hook(self.ablation.get_hook_ablate(idx, mask))

        for idx in tqdm(range(len(self.mmtom))):
            logsm = self.compute_row(idx)
            result_list.append(logsm)
        return result_list
    
    def extract_highest_val(self, row):
        return max(row, key=lambda x: x[1])[0].strip().lower()
    
    def experiment(self, pct: float=0.01):
        self.ablation.clear_hooks(self.vlm)
        self.vlm.model.eval()
        assess_dict = OrderedDict([
            ("no_ablation", None),
            (f"ablate_top", self.loc_units.get_masked_ktop(pct).T),
            (f"ablate_random1", self.loc_units.get_random_mask(pct, seed=42).T),
            (f"ablate_random2", self.loc_units.get_random_mask(pct, seed=12345).T),
            (f"ablate_random3", self.loc_units.get_random_mask(pct, seed=98765).T),
            (f"ablate_random4", self.loc_units.get_random_mask(pct, seed=2024).T),
            (f"ablate_random5", self.loc_units.get_random_mask(pct, seed=56789).T),
            (f"ablate_random6", self.loc_units.get_random_mask(pct, seed=2025).T),
        ])

        info_process = ["No Ablation",
                        f"Ablation Top-{pct*100:.2f}%",
                        f"Random Ablation {pct*100:.2f}% (1/6)",
                        f"Random Ablation {pct*100:.2f}% (2/6)",
                        f"Random Ablation {pct*100:.2f}% (3/6)",
                        f"Random Ablation {pct*100:.2f}% (4/6)",
                        f"Random Ablation {pct*100:.2f}% (5/6)",
                        f"Random Ablation {pct*100:.2f}% (6/6)",
                        ]
        
        data = self.mmtom.df.copy()
        for idx, (key, mask) in islice(enumerate(assess_dict.items()), 0, None):
            # Display progress
            print(f"Step {idx+1}/{len(info_process)}: {info_process[idx]}")
            result_list = self.compute(mask)
            data[f"logsm_{key}"] = result_list
            data[f"predict_{key}"] = data[f"logsm_{key}"].apply(self.extract_highest_val)
        
        return data
