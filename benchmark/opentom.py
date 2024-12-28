import pandas as pd
from .utils import BenchmarkText
from typing import Optional

class BenchmarkOpenToM(BenchmarkText):
    def __init__(self, 
                 story: Optional[str] = None,
                 order: Optional[str]="first_order",
                 subset: Optional[int]=None):
        """ OpenToM Benchmark"""
        super().__init__()
        if story:
            self.story = story
        else:
            self.story = "narrative"
        df = self.preprocess()
        df = self.process_opentom(df)
        self.data = self.intermediate_prompting(df, order, subset)
        self.expanded_df = self.build_expanded_df()
    
    def process_opentom(self, df):
        """ Preprocessing of the result of OpenToM or Variant_OpenToM """
        
        # Identify tertiary questions
        df["is_tertiary"] = df['answer'].str.startswith(('less', 'equally', 'more'))
        
        # Assign Qtype based on conditions
        df["Qtype"] = df.apply(
            lambda row: "[Yes, No]" if row["is_closed_question"]
            else "[Less, Equally, More]" if row["is_tertiary"]
            else "[Initial loc., New loc.]", 
            axis=1
        )
        return df

    def classic_prompting(self, 
                          df: pd.DataFrame,
                          order: Optional[str]="first_order",
                          subset: Optional[int]=None):
        context = (
            "The following multiple choice question is based on the following story. The question "
            "is related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer "
            "from the options provided by printing it as is without any modifications."
        )

        df["prompt"] = df.apply( lambda row: f"{context}\nStory: {row[self.story]}\nQuestion: {row['question']}\nOptions:\n" + 
                        "\n".join([f"- {cand}" for cand in row["cands"]]) +
                        "\nAnswer:", axis=1)
        
        # Remove the two rows with incorrect values
        # df.drop([4884, 4904, 10248, 10268], inplace=True)

        # Keep only False Belief Stories
        df = df[~df["observed"]].copy()


        if (order is not None):
            df = df[df["qOrder"]==order]

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        df.reset_index(drop=True, inplace=True)
    
        return df 

    def intermediate_prompting(self, 
                          df: pd.DataFrame,
                          order: Optional[str]=None,
                          subset: Optional[int]=None):
        
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
                f"{context}\nStory: {row[self.story]}\nQuestion: {row['question']}\n"
                f"Options:\n{enumerated_cands}\nAnswer:"
            )

        # Apply the function to generate prompts
        df["prompt"] = df.apply(generate_prompt, axis=1)

        # Remove the two rows with incorrect values
        # df.drop([4884, 4904, 10248, 10268], inplace=True)

        # Keep only False Belief Stories
        df = df[~df["observed"]].copy().reset_index(drop=True)

        if (order is not None):
            df = df[df["qOrder"]==order]
            df.reset_index(drop=True, inplace=True)

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
    
        return df 

    def preprocess(self):
        """
        I only consider the Long split. 
        
        """
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
            [row["plot_info"]["original_place"], row["plot_info"]["move_to_place"]] if row["qOrder"] in ["first_order", "second_order"] else 
            ["positive", "negative", "neutral"]),
            axis=1)
        
        # Remove the two rows with incorrect values
        df.drop([4884, 4904, 10248, 10268], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def __getitem__(self, idx):
        return {"text": self.data["prompt"].iloc[idx]}