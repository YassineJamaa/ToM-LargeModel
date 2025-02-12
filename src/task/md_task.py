import numpy as np
from torch.utils.data import Dataset

class MDLocDataset(Dataset):
    def __init__(self):  
        num_examples = 100

        self.positive = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(100, 200)
            num_2 = np.random.randint(100, 200)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.positive.append(f"Question: {question}\nAnswer: {answer}")

        self.negative = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(1, 20)
            num_2 = np.random.randint(1, 20)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.negative.append(f"Question: {question}\nAnswer: {answer}")

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()
        
    def __len__(self):
        return len(self.positive) 