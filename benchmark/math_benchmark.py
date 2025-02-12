import random
from benchmark import BenchmarkText
import pandas as pd
import os
import json

context = (
    "This is a mathematical problem."
    " Read it and choose the best answer among options provided: a, b, c, d."
)

class MATHBenchmark(BenchmarkText):
    def __init__(self,
                 context: str=context, 
                 level: int=None,
                 type_: str=None,
                 subset: int=None):
        self.level = level
        self.type = type_
        self.subset = subset
        df = self.preprocessing(context)
        df = self.filtering(df)
        self.data = df

    def filtering(self, df: pd.DataFrame):
        if self.type:
            df = df.loc[df["Type"]==self.type]
        
        if self.level:
            level_name = f"Level {self.level}"
            df = df.loc[df["Level"]==level_name]
        
        df.reset_index(drop=True, inplace=True)
        
        if self.subset:
            df = df.loc[:self.subset,:]
        return df
    
    def preprocessing(self, context):
        json_math = os.path.join("dataset", "benchmarks", "MATH", "test.jsonl")
        data = []
        with open(json_math, 'r') as file:
            for line in file:
                data.append(json.loads(line))
                
        df = pd.DataFrame(data)
        # context = (
        #     "This is a mathematical problem."
        #     "Choose the best answer only among options provided."
        #     "After 'Answer :' you should respond the best option among: a, b, c, d."
        # )
        df['cands'] = df[['A', 'B', 'C', 'D']].values.tolist()
        df['cands_letter'] = df['cands'].apply(lambda x: [f"{chr(97+i)}" for i, opt in enumerate(x)])
        df["answer_letter"] = df["Answer"].apply(lambda x: x.lower())
        df['answer'] = df.apply(lambda row: row[row['Answer']], axis=1)
        df['prompt'] = df.apply(lambda row: f"{context}\nQuestion: {row['Question']}\nOptions:\n" + "\n".join([f"{chr(97+i)}. {opt}" for i, opt in enumerate(row['cands'])]) + "\nAnswer letter:", axis=1)
        return df
    
    def __getitem__(self, idx):
        return {"text": self.data["prompt"].iloc[idx]}