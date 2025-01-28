from dotenv import load_dotenv
import os

def setup_environment(cache="CACHE_DIR"):
    """Load environment variables and configure device."""
    load_dotenv()  # Ensure .env is in the project root
    hf_access_token = os.getenv("HF_ACCESS_TOKEN")
    cache_dir = os.getenv(cache)
    return hf_access_token, cache_dir

def load_chat_template(file_path: str):
    # Load the chat_template from the file
    with open(file_path, "r") as file:
        chat_template = file.read()
    return chat_template
