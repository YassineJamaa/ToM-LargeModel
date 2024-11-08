def process_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        
    # Split paragraphs by double spaces
    paragraphs = content.split('\n\n')
    
    # Check that we have exactly three parts: Story, Question, Answer
    if len(paragraphs) == 3:
        story = paragraphs[0].strip()
        question = paragraphs[1].strip()
        answer = paragraphs[2].strip()
        
        # Format according to the new structure
        formatted_text = (
            "In this experiment, you will read a series of sentences and then answer True/False.\n"
            "Press button 1 to answer 'true' and button 2 to answer 'false'.\n\n"
            f"Story: {story}\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n"
        )
        
        return formatted_text
    else:
        return "Error: The file does not contain the expected format."

