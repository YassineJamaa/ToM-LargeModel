from typing import Optional
import pandas as pd
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from .utils import BenchmarkVisionText

class BenchmarkMMToMQA(BenchmarkVisionText):
    def __init__(self, subset:Optional[int]=None):
        super().__init__()
        # Directory of videos
        self.dir_videos = "dataset/benchmarks/MMToMQA/videos/single_agent_partial_train_240_hp_test_highres"
        self.dir_text = "dataset/benchmarks/MMToMQA/text/questions.json"

        # Question Type
        self.q_type = {
            "True belief": 1.1,
            "False belief": 1.2,
            "Belief tracking": 1.3,
            "Goal inference true belief": 2.1,
            "Goal inference false belief": 2.2,
            "Goal inference updated belief": 2.3,
            "Goal inference future actions": 2.4
        }

        # Extract text dataset
        df = pd.read_json(self.dir_text, lines=True)
        # self.data = df[df["question_type"]!=2.4].copy()
        self.data = df.copy()
        if subset is not None:
            self.data = self.data.iloc[:subset].copy()
        
        # Reset Index
        self.data.reset_index(inplace=True)

        # Rename columns
        self.data.rename(columns={"answer_list": "cands", "question": "prompt"}, inplace=True)

    def __len__(self):
        return len(self.data)
    
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
        return prompt, frames
    
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