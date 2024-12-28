import pandas as pd
from torch.utils.data import Dataset

class BenchmarkBaseline(Dataset):
    def __init__(self):
        self.data = None

    def set_candidates(self, benchmark_df: pd.DataFrame):
        if "cands" not in benchmark_df.columns or "answer" not in benchmark_df.columns:
            raise ValueError("The DataFrame must contain 'cands' and 'answer' columns.")
        
        def generate_tokens(candidates):
            num_candidates = len(candidates)
            letters = [chr(97 + i) for i in range(num_candidates)]  # Generate letters a, b, c, ...
            uppercase = [f"{letter.upper()}" for letter in letters]
            lowercase = [f"{letter}" for letter in letters]
            # Flatten into a 1D list
            return lowercase + uppercase

        def map_answer_to_letter(candidates, answer):
            letters = [chr(97 + i) for i in range(len(candidates))]  # Generate letters a, b, c, ...
            answer_index = candidates.index(answer)  # Get index of the correct answer
            return letters[answer_index]  # Return the corresponding letter
        
        benchmark_df["cands_letter"] = benchmark_df["cands"].apply(generate_tokens)
        benchmark_df["answer_letter"] = benchmark_df.apply(
            lambda row: map_answer_to_letter(row["cands"], row["answer"]), axis=1
        )
        return benchmark_df
    
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


    
    
