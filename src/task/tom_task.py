from torch.utils.data import Dataset
import numpy as np

def append_options(question):
    options = ["True", "False"]
    return f"{question}\nOptions:\n- {options[0]}\n- {options[1]}"

def read_story(path):
    with open(path, "r") as f:
        story = [line.strip() for line in f.readlines()]
    return ' '.join(story)

def read_question(path):
    with open(path, "r") as f:
        question = f.read().split('\n\n')[0]
    return append_options(question.replace("\n", ""))

class ToMLocDataset(Dataset):
    def __init__(self):
        instruction = "In this experiment, you will read a series of sentences and then answer True/False questions about them. Press button 1 to answer 'true' and button 2 to answer 'false'."
        context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer: {answer}"
        belief_stories = [read_story(f"dataset/localizer/tomloc/{idx}b_story.txt") for idx in range(1, 11)]
        photograph_stories = [read_story(f"dataset/localizer/tomloc/{idx}p_story.txt") for idx in range(1, 11)]

        belief_question = [read_question(f"dataset/localizer/tomloc/{idx}b_question.txt") for idx in range(1, 11)]
        photograph_question = [read_question(f"dataset/localizer/tomloc/{idx}p_question.txt") for idx in range(1, 11)]

        # Set a fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)
        
        self.positive = [context_template.format(instruction=instruction, story=story, question=question, answer=rng.choice(["True", "False"])) for story, question in zip(belief_stories, belief_question)]
        self.negative = [context_template.format(instruction=instruction, story=story, question=question, answer=rng.choice(["True", "False"])) for story, question in zip(photograph_stories, photograph_question)]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()
    
    def __len__(self):
        return len(self.positive)


