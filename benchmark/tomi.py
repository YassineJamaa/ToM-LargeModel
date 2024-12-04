import pandas as pd
import ast
from .utils import BenchmarkBaseline
from typing import Optional

class BenchmarkToMi(BenchmarkBaseline):
    def __init__(self, subset: Optional[int]=None):
        super().__init__()
        
        self.data = self.classic_prompting(subset=subset)
        self.expanded_df = self.build_expanded_df()
    
    def classic_prompting(self, subset):
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

        # Keep only the False Belief Stories
        df = df[~df["falseTrueBelief"]].copy()

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        return df

    # def __len__(self):
    #     """
    #     Returns the total number of samples in the dataset.
    #     """
    #     return len(self.expanded_df)