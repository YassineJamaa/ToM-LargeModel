import pandas as pd
from torch.utils.data import Dataset

class BenchmarkBaseline(Dataset):
    def __init__(self):
        self.data = None
    
    def build_expanded_df(self):
        return pd.DataFrame({
            "id_prompt": self.data.index.repeat(self.data["cands"].str.len()),
            "id_cand": [i for sublist in self.data["cands"] for i in range(len(sublist))],
            "cand": [cand for cands in self.data["cands"] for cand in cands],
            "fusion": [f"{prompt} {cand}" for prompt, cands in zip(self.data["prompt"], self.data["cands"]) for cand in cands],
            "prompt_end_pos": [
                len(prompt) for prompt, cands in zip(self.data["prompt"], self.data["cands"]) for _ in cands
            ]
        })
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    

class BenchmarkText(BenchmarkBaseline):
    def __init__(self):
        super().__init__()
        self.expanded_df = None
    
    def build_expanded_df(self):
        return pd.DataFrame({
            "id_prompt": self.data.index.repeat(self.data["cands"].str.len()),
            "id_cand": [i for sublist in self.data["cands"] for i in range(len(sublist))],
            "cand": [cand for cands in self.data["cands"] for cand in cands],
            "fusion": [f"{prompt} {cand}" for prompt, cands in zip(self.data["prompt"], self.data["cands"]) for cand in cands],
            "prompt_end_pos": [
                len(prompt) for prompt, cands in zip(self.data["prompt"], self.data["cands"]) for _ in cands
            ]
        })

class BenchmarkVisionText(BenchmarkBaseline):
    def __init__(self):
        super().__init__()


    
    
