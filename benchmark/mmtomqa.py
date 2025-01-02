from typing import Optional
import pandas as pd
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from benchmark import BenchmarkVisionText

# Question Type
q_type = {
    "True belief": 1.1,
    "False belief": 1.2,
    "Belief tracking": 1.3,
    "Goal inference true belief": 2.1,
    "Goal inference false belief": 2.2,
    "Goal inference updated belief": 2.3,
    "Goal inference future actions": 2.4
}
q_type_invert = {value: key for key, value in q_type.items()}

class BenchmarkMMToMQA(BenchmarkVisionText):
    def __init__(self, subset:Optional[int]=None):
        super().__init__()
        # Directory of videos
        self.dir_videos = "dataset/benchmarks/MMToMQA/videos/single_agent_partial_train_240_hp_test_highres"
        self.dir_text = "dataset/benchmarks/MMToMQA/text/questions.json"

        # Extract text dataset
        df = pd.read_json(self.dir_text, lines=True)

        # Choose the question type
        # self.data = df[df["question_type"]!=2.4].copy()
        self.data = df.copy()
        if subset is not None:
            self.data = self.data.iloc[:subset].copy()
        
        # Reset Index
        self.data.reset_index(inplace=True)

        # Rename columns
        self.data.rename(columns={"answer_list": "cands", "question": "prompt"}, inplace=True)

        # Prompting
        self.data = self.intermediate_prompting(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def process_df(self, df: pd.DataFrame):

        #Extract candidates
        target_phrase = "which one of the following statements is more likely to be true?"
        df['candidates'] = df['prompt'].str.extract(f'(?i){target_phrase}\\?\\s*(.*)', expand=True)
        df['candidates'] = df['candidates'].str.replace(" Please respond with either a or b.", "", regex=False)
        df['cands'] = df['candidates'].str.findall(r'\(\w\)\s*[^()]+')
        df['cands_letter'] = df['candidates'].str.findall(r'\(([a-zA-Z])\)')  # Letters including upper and lower case
        df['cands_letter'] = df['cands_letter'].apply(lambda letters: letters + [letter.upper() for letter in letters if letter.islower()])
        
        # Extract Story
        df['story'] = df['prompt'].str.extract(r'(.*)\s+Question:', expand=True)
        df['story'] = df['story'].str.strip()

        # Extract Question
        df['question'] = df['prompt'].str.extract(r'Question:\s*(.*?)\s*\(a\)', expand=True)

        return df
    
    def intermediate_prompting(self, df: pd.DataFrame):
        # Process DataFrame
        self.data = self.process_df(self.data)
        df.rename(columns={"prompt":"preprompt"}, inplace=True)

        # Contexts
        context = (
            "The following multiple choice question is based on the following story. The question "
            "is related to Theory-of-Mind. Read the story and then answer the question. Choose the best answer "
            "among options provided."
        )

        # Dynamically construct the prompts
        def generate_prompt(row):
            # Dynamically enumerate candidates
            enumerated_cands = "\n".join([f"{cand}" for i, cand in enumerate(row["cands"])])
            return (
                f"{context}\nStory:{row['story']}\nQuestion: {row['question']}\n"
                f"Options:\n{enumerated_cands}\nAnswer:"
            )

        # Apply the function to generate prompts
        df["prompt"] = df.apply(generate_prompt, axis=1)
        df.rename(columns={"answer":"answer_letter"}, inplace=True)

        # Map the Qtype
        df["Qtype"] = df["question_type"].map(q_type_invert)
        # df = df[df["question_type"].isin([1.2, 2.2])].copy()

        return df
    
    def get_frames(self, episode, selected_frames):
        # Path for the episode
        folder_path = os.path.join(self.dir_videos, f"task_{episode}", "script", "0")

        # Path the selected frames
        selected_files = [f"{folder_path}/Action_{frame}_0_normal.png" for frame in selected_frames]

        # Loading the selected files
        loaded_frames = [Image.open(selected_file) for selected_file in selected_files]

        return loaded_frames

    def extract_middle_frame(self, pos):
        """ Extract the prompt """
        episode = self.data["episode"].iloc[pos]
        start_time = self.data["start_time"].iloc[pos]
        end_time = self.data["end_time"].iloc[pos]
        path_intervals = os.path.join(self.dir_videos, f"task_{episode}", "frame_intervals.pik")

        # Open the .pik file in binary read mode
        with open(path_intervals, 'rb') as file:
            frame_intervals = pickle.load(file)

        # Get the middle position for each interval
        mid_frames = [int((start + end)/2) for start, end in frame_intervals]

        # List of desired frame indices
        selected_frames = [f"{frame:04d}" for index, frame in enumerate(mid_frames) if start_time <= index <= end_time]

        return self.get_frames(episode, selected_frames)

    def __getitem__(self, pos):
        """ Extract the prompt with the corresponding videos frames """
        frames = self.extract_middle_frame(pos)
        prompt = self.data["prompt"].iloc[pos]
        return {"text": prompt, "frames": frames}
    
    def plot_text_frames(self, pos):
        prompt, frames = self[pos]

        # Story
        print(f"{prompt}\n")

        # Check the number of frames loaded
        print(f"Loaded {len(frames)} frames.")

        # Display the frames using matplotlib
        for idx, frame in enumerate(frames):
            plt.figure(figsize=(8, 8))  # Adjust the figure size
            plt.imshow(frame)
            plt.axis('off')  # Hide axes
            plt.title(f"Frame {idx}")  # Add a title with the frame index
            plt.show()