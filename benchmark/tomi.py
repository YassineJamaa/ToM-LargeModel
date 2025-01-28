import pandas as pd
import ast
from .utils import BenchmarkText
from typing import Optional

class BenchmarkToMi(BenchmarkText):
    def __init__(self, subset: Optional[int]=None):
        super().__init__()
        
        self.data = self.intermediate_prompting(subset=subset)
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
            "among options provided."
        )

        df["prompt"] = df.apply(
            lambda row: f"{context}\nStory: {row["story"]}\nQuestion: {row["question"]}\nOptions:\n- {row["cands"][0]}\n- {row["cands"][1]}\nAnswer:",
            axis=1
        )

        # Keep only the False Belief Stories
        df = df[~df["falseTrueBelief"]].copy().reset_index(drop=True)

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        return df
    
    def intermediate_prompting(self, subset):
        csv_file = "dataset/benchmarks/ToMi/ToMi-finalNeuralTOM.csv"
        df = pd.read_csv(csv_file)
        df = df[df["qOrder"] == 'first_order'].reset_index(drop=True)

        # Convert the 'cands' column from string representation of lists to actual lists
        df["cands"] = df["cands"].apply(ast.literal_eval)

        df = self.set_candidates(df)

        # Contexts
        context = (
            "The following multiple choice question is based on the following story. The question "
            "is related to Theory-of-Mind. Read the story and then answer the question. Choose the best answer "
            "among options provided."
        )

        # Dynamically construct the prompts
        def generate_prompt(row):
            # Dynamically enumerate candidates
            enumerated_cands = "\n".join([f"({chr(97 + i)}) {cand}" for i, cand in enumerate(row["cands"])])
            return (
                f"{context}\nStory:{row['story']}\nQuestion: {row['question']}\n"
                f"Options:\n{enumerated_cands}\nAnswer:"
            )

        # Apply the function to generate prompts
        df["prompt"] = df.apply(generate_prompt, axis=1)

        # Keep only the False Belief Stories
        df = df[~df["falseTrueBelief"]].copy().reset_index(drop=True)

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        return df

    def __getitem__(self, idx):
        return {"text": self.data["prompt"].iloc[idx]}


class ToMiPromptEngineering(BenchmarkText):
    def __init__(self, subset: Optional[int]=None):
        super().__init__()
        
        self.data = self.intermediate_prompting(subset=subset)
        self.expanded_df = self.build_expanded_df()
    
    def classic_prompting(self, subset):
        csv_file="dataset/benchmarks/ToMi/ToMi-finalNeuralTOM.csv"
        df = pd.read_csv(csv_file)
        df = df[df["qOrder"]=='first_order'].reset_index(drop=True)
        # Convert the 'cands' column from string representation of lists to actual lists
        df["cands"] = df["cands"].apply(ast.literal_eval)

        context = (
            "The following multiple choice question is based on the following story. The question "
            "is related to Theory-of-Mind. Read the story and then answer the question. You are tasked to make "
            " a decision from the given options: a, b. Respond with only one letter that represents your decision."
            "Do not include any additional text, explanation, or punctuation—just the single letter corresponding to your choice."
        )

        df["prompt"] = df.apply(
            lambda row: f"{context}\nStory: {row["story"]}\nQuestion: {row["question"]}\nOptions:\n- {row["cands"][0]}\n- {row["cands"][1]}\nAnswer:",
            axis=1
        )

        # Keep only the False Belief Stories
        df = df[~df["falseTrueBelief"]].copy().reset_index(drop=True)

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        return df
    
    def intermediate_prompting(self, subset):
        csv_file = "dataset/benchmarks/ToMi/ToMi-finalNeuralTOM.csv"
        df = pd.read_csv(csv_file)
        df = df[df["qOrder"] == 'first_order'].reset_index(drop=True)

        # Convert the 'cands' column from string representation of lists to actual lists
        df["cands"] = df["cands"].apply(ast.literal_eval)

        df = self.set_candidates(df)

        # Contexts
        # context = (
        #     "The following multiple choice question is based on the following story. The question "
        #     "is related to Theory-of-Mind. Read the story and then answer the questions. You are tasked to make "
        #     " a decision from the given options: a, b. Respond with only one letter that represents your decision."
        #     "Do not include any additional text, explanation, or punctuation—just the single letter corresponding to your choice."
        # )

        context = (
            "The following multiple choice question is based on the following story. The question "
            "is related to Theory-of-Mind. Read the story and then answer the question. Choose the best answer "
            "among options provided."
        )

        # Dynamically construct the prompts
        def generate_prompt(row):
            # Dynamically enumerate candidates
            enumerated_cands = "\n".join([f"{chr(97 + i)}. {cand}" for i, cand in enumerate(row["cands"])])
            return (
                f"{context}\nStory:{row['story']}\nQuestion: {row['question']}\n"
                f"Options:\n{enumerated_cands}\nAnswer:"
            )

        # Apply the function to generate prompts
        df["prompt"] = df.apply(generate_prompt, axis=1)

        # Keep only the False Belief Stories
        df = df[~df["falseTrueBelief"]].copy().reset_index(drop=True)

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        return df

    def __getitem__(self, idx):
        return {"text": self.data["prompt"].iloc[idx]}