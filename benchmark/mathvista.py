from datasets import load_dataset
import numpy as np
import pandas as pd
from .utils import BenchmarkText

class MathVistaBenchmark(BenchmarkText): # Change the inheritance to BenchmarkText
    def __init__(self,
                 subset: int=None):
        df = self.preprocessing()
        self.data = self.prompting(df, subset)

    def preprocessing(self):
        ds = load_dataset("AI4Math/MathVista") # Change the cache_dir
        df = pd.DataFrame(ds["testmini"])
        df = df.loc[df["choices"].notna()]
        df.rename(columns={"choices": "cands"}, inplace=True)
        return df
    
    def prompting(self, df: pd.DataFrame, subset: int=None):
        context = (
            "Given an image and a related question with multiple-choice options, "
            "analyze the image and select the best answer from the provided choices:"
        )
        df = self.set_candidates(df)

        df["context_options"] = df["cands_letter"].apply(lambda x: f"{context} {", ".join(sorted(set(x)))}")

        # Dynamically construct the prompts
        def generate_prompt(row):
            # Dynamically enumerate candidates
            enumerated_cands = "\n".join([f"{chr(97 + i)}. {cand}" for i, cand in enumerate(row["cands"])])
            return (
                f"{row["context_options"]}\nQuestion: {row['question']}\n"
                f"Options:\n{enumerated_cands}\nAnswer:"
            )
        
        # Apply the function to generate prompts
        df["prompt"] = df.apply(generate_prompt, axis=1)
        
        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]

        return df
    
    def __getitem__(self, pos):
        """ Extract the prompt with the corresponding videos frames """
        frames = self.data["decoded_image"].iloc[pos]
        prompt = self.data["prompt"].iloc[pos]
        return {"text": prompt, "frames": frames}
