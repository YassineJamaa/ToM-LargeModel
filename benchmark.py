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
    
class BenchmarkOpenToM(Dataset):
    def __init__(self, 
                 order: Optional[str]=None,
                 subset: Optional[int]=None):
        df = self.preprocess()
        self.data = self.classic_prompting(df, order, subset)
        self.expanded_df = pd.DataFrame({
            "id_prompt": self.data.index.repeat(self.data["cands"].str.len()),
            "id_cand": [i for sublist in self.data["cands"] for i in range(len(sublist))],
            "cand": [cand for cands in self.data["cands"] for cand in cands],
            "fusion": [f"{prompt} {cand}" for prompt, cands in zip(self.data["prompt"], self.data["cands"]) for cand in cands],
            "prompt_end_pos": [
                len(prompt) for prompt, cands in zip(self.data["prompt"], self.data["cands"]) for _ in cands
            ]
        })

    def classic_prompting(self, 
                          df: pd.DataFrame,
                          order: Optional[str]=None,
                          subset: Optional[int]=None):
        context = (
            "The following multiple choice question is based on the following story. The question "
            "is related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer "
            "from the options provided by printing it as is without any modifications."
        )

        df["prompt"] = df.apply( lambda row: f"{context}\nStory: {row['narrative']}\nQuestion: {row['question']}\nOptions:\n" + 
                        "\n".join([f"- {cand}" for cand in row["cands"]]) +
                        "\nAnswer:", axis=1)
        
        if (order is not None):
            df = df[df["qOrder"]==order]
            df.reset_index(drop=True, inplace=True)

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        return df  

    def preprocess(self):
        """ For now, I only consider the Long split (need to adapt the code for taking also 'ExtraLong') """
        splits = {'Long': 'opentom.json', 'ExtraLong': 'opentom_long.json'}
        df = pd.read_json("hf://datasets/SeacowX/OpenToM/" + splits["Long"])
        df["question_text"] = df["question"].apply(lambda x: x["question"])
        df["answer"] = df["question"].apply(lambda x: x["answer"])
        df["type"] = df["question"].apply(lambda x: x["type"])
        df = df.drop(columns=["question"])
        df.rename(columns={"question_text": "question"}, inplace=True)
        df["is_closed_question"] = df["answer"].apply(lambda x: True if x in ["Yes", "No"] else False)
        df["qOrder"] = df["type"].apply(lambda x: "first_order" if x in ["location-fo", "multihop-fo"] 
                        else "second_order" if x in ["location-so", "multihop-so"] 
                        else "No")
        df["cands"] = df.apply(
            lambda row: ["Yes", "No"] if row["is_closed_question"] else 
            (["less " + row["answer"].split()[1], "equally "+row["answer"].split()[1], "more "+row["answer"].split()[1]] if row["type"].startswith("multihop") else 
            [row["plot_info"]["original_place"], row["plot_info"]["move_to_place"]] if row["qOrder"] in ["first order", "second order"] else 
            ["positive", "negative", "neutral"]),
            axis=1)
        return df
    
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