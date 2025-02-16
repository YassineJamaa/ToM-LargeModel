import pandas as pd
import numpy as np
import io
from PIL import Image
import re
from .utils import BenchmarkText

class MMStarBenchmark(BenchmarkText):
    
    def __init__(self, subset: int=None):  
        super().__init__()
        df = self.preprocessing()
        self.data = self.prompting(df, subset)


    def prompting(self, df: pd.DataFrame, subset:int=None):
        df = self.set_candidates(df)
        context = (
            "Given an image and a related question with multiple-choice options, "
            "analyze the image and select the best answer from the provided choices:"
        )
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
    
    def preprocessing(self):
        df = pd.read_parquet("hf://datasets/Lin-Chen/MMStar/mmstar.parquet")

        # Convert raw byte image into PIL image
        df.rename(columns={"image": "raw_byte_img"}, inplace=True)
        df["image"] = df["raw_byte_img"].apply(lambda x: Image.open(io.BytesIO(x)))

        # Extract cands from question and extract the question
        df.rename(columns={"question": "raw_question"}, inplace=True)
        df["question"] = df["raw_question"].apply(lambda x: x.split("\nOptions:")[0])

        # Extract options from question_raw
        def extract_candidates(question):
            # Try to match explicit labels (Options or Choices)
            match = re.search(r"(?:Options|Choices):\s*(.*)", question, re.DOTALL)

            # If no explicit label is found, assume everything after "Question:" contains the options
            if not match:
                match = re.search(r"Question:.*?\n([\s\S]*)", question, re.DOTALL)

            if not match:
                return []

            options_text = match.group(1)
            candidates = []
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # List of possible option letters
            i = 0

            while i < len(letters):
                # Search for the next option starting with a letter formatted as (A) or A:
                pattern = rf"\({letters[i]}\)|{letters[i]}:"
                match = re.search(pattern, options_text)
                
                if match:
                    start = match.end()  # Start of the option content
                    next_match = None

                    # Look for the next option to determine where this option ends
                    for j in range(i + 1, len(letters)):
                        next_pattern = rf"\({letters[j]}\)|{letters[j]}:"
                        next_match = re.search(next_pattern, options_text[start:])
                        if next_match:
                            end = start + next_match.start()
                            break
                    else:
                        end = len(options_text)  # If no next option is found, take until the end

                    option_text = options_text[start:end].strip().rstrip(',')
                    candidates.append(option_text)

                i += 1  # Move to the next letter

            return candidates
        df["cands"] = df["raw_question"].apply(extract_candidates)

        # Convert the Answer which the answer letter into the corresponding candidate text
        def replace_answer(row):
            if isinstance(row["answer"], str) and row["answer"].isalpha():  # Ensure valid letter
                index = ord(row["answer"]) - ord("A")  # Convert letter to index (A -> 0, B -> 1, etc.)
                return row["cands"][index] if 0 <= index < len(row["cands"]) else row["answer"]
            return row["answer"]  # Return original value if there's an issue

        # Replace answers with actual responses
        df["test"] = df.apply(replace_answer, axis=1)
        df.rename(columns={"answer": "raw_answer", "test":"answer"}, inplace=True)
        return df
    
    def __getitem__(self, pos):
        """ Extract the prompt with the corresponding videos frames """
        frames = self.data["image"].iloc[pos]
        prompt = self.data["prompt"].iloc[pos]
        return {"text": prompt, "frames": frames}

mmstar = MMStarBenchmark()        