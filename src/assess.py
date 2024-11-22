from .localise_units import ImportLLMfromHF, LocImportantUnits
from .benchmark import BenchmarkToMi, BenchmarkOpenToM
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from typing import Optional
from torch.utils.data import Dataset
from collections import OrderedDict
import os
from itertools import islice

def compute_cand_score(row):
    logits = row['log_sm']
    n = row['last_tokens']
    last_token = row["last_token_position"]
    if n == 1:
        return logits[last_token-1] # Return the last element
    elif n > 1:
        return sum(logits[last_token-n:last_token-1]) / n  # Return the mean of the last N elements

class AssessBenchmark:
    def __init__(self,
                 llm: ImportLLMfromHF,
                 loc_units: LocImportantUnits):
        self.llm = llm
        self.loc_units = loc_units
    
    def clear_hooks(self):
        """Clears all registered forward hooks in the model layers."""
        for layer in self.llm.model.model.layers:
            layer._forward_hooks.clear()
    
    def get_hook_ablate(self, idx, mask):
        """
        Defines a hook function to ablate specific units based on a mask.
        
        Args:
            idx (int): Layer index.
            mask (torch.Tensor): Binary mask for ablation at the given layer.

        Returns:
            function: A hook function to zero out specified units.
        """
        def hook_ablate(module, input, output):
            mask_layer = mask[idx]
            unit_indices = mask_layer.nonzero()[0]
            output[0][:, :, unit_indices] = 0
        return hook_ablate
    
    def assess(self, 
           data: pd.DataFrame,
           mask: Optional[np.ndarray] = None,
           batch_size: int = 20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.model.to(device)
        self.clear_hooks()

        if mask is not None:
            for idx, layer in enumerate(self.llm.model.model.layers):
                layer.register_forward_hook(self.get_hook_ablate(idx, mask))

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
                   bn_data: Dataset,
                   check_path: Optional[str]=None,
                   batch_size: int = 20,
                   pct=0.01):
        self.llm.model.eval()
        assess_dict = OrderedDict([
            ("no_ablation", None),
            (f"ablate_top_{int(pct * 100)}", self.loc_units.get_masked_ktop(pct).T),
            (f"ablate_random1_{int(pct * 100)}", self.loc_units.get_random_mask(pct, seed=42).T),
            (f"ablate_random2_{int(pct * 100)}", self.loc_units.get_random_mask(pct, seed=12345).T),
            (f"ablate_random3_{int(pct * 100)}", self.loc_units.get_random_mask(pct, seed=98765).T)
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