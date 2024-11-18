import ast
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional

class BenchmarkToMi(Dataset):
    def __init__(self, subset: Optional[int]=None):
        csv_file="dataset/benchmarks/ToMi/ToMi-finalNeuralTOM.csv"
        df = pd.read_csv(csv_file)
        df = df[df["qOrder"]=='first_order'].reset_index(drop=True)
        # Convert the 'cands' column from string representation of lists to actual lists
        df["cands"] = df["cands"].apply(ast.literal_eval)
        context = (
            "The following multiple choice question is based on the following story. The question "
            "is related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer "
            "from the options provided by printing it as is without any modifications."
        )

        df["prompt"] = df.apply(
            lambda row: f"{context}\nStory: {row["story"]}\nQuestion: {row["question"]}\nOptions:\n- {row["cands"][0]}\n- {row["cands"][1]}\nAnswer:",
            axis=1
        )

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        self.data = df
        self.expanded_df = pd.DataFrame({
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
        return len(self.expanded_df)

    def __getitem__(self, idx):
        """
        Fetches a single data point from the dataset.
        
        Args:
            idx (int): Index of the data point.
        
        Returns:
            dict: A dictionary containing the data for the specified index.
        """
        row = self.expanded_df.iloc[idx]
        return {
            "id_prompt": row["id_prompt"],
            "id_cand": row["id_cand"],
            "cand": row["cand"],
            "fusion": row["fusion"],
            "prompt_end_pos": row["prompt_end_pos"]
        }
    
