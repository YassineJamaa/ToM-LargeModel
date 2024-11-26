from torch.utils.data import Dataset
import numpy as np

class ExtendedTomLocGPT4(Dataset):
    def __init__(self):
        instruction = "In this experiment, you will read a series of sentences and then answer True/False questions about them. Press button 1 to answer 'true' and button 2 to answer 'false'."
        context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer: {answer}"

        # Import Photo Stories and Questions
        photo_path = 'dataset/localizer/extended-tomloc/tomloc_negative.txt'
        photograph_stories, photograph_questions = self.extract_stories_and_questions(photo_path)

        # Import Belief Stories and Questions
        belief_path = 'dataset/localizer/extended-tomloc/tomloc_positive.txt'
        belief_stories, belief_questions = self.extract_stories_and_questions(belief_path)

        # Set a fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)
        
        # Generate positive and negative examples
        self.positive = [
            context_template.format(
                instruction=instruction,
                story=story,
                question=question,
                answer=rng.choice(["True", "False"])
            )
            for story, question in zip(belief_stories, belief_questions)
        ]

        self.negative = [
            context_template.format(
                instruction=instruction,
                story=story,
                question=question,
                answer=rng.choice(["True", "False"])
            )
            for story, question in zip(photograph_stories, photograph_questions)
        ]
        
        # 100 Photo Stories vs 95 Belief Stories
        self.negative = self.negative[:95]
   
    def extract_stories_and_questions(self, file_path):
        # Initialize lists to store stories and questions
        stories = []
        questions = []

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            content = file.read()

        # Split the content by double newlines to separate each story-question block
        blocks = content.split('\n\n')

        # Process each block to extract the story and question
        for block in blocks:
            if block.startswith("Story:") and "Question:" in block:
                # Split the block into story and question
                story_part, question_part = block.split("Question:", 1)
                # Clean up and extract the text for story and question
                story = story_part.replace("Story:", "").strip()
                question = question_part.strip()
                # Append to respective lists
                stories.append(story)
                questions.append(question)

        return stories, questions
    
    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()
    
    def __len__(self):
        return len(self.positive)