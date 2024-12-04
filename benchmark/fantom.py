import pandas as pd
from .utils import BenchmarkBaseline
from typing import Optional

import os
import requests
import hashlib
import zipfile
import tarfile
import datetime
from tqdm import tqdm
import pandas as pd
import argparse

class DownloadableFile:
    def __init__(self, url, filename, expected_hash, version="1.0", zipped=True):
        self.url = url
        self.filename = filename
        self.expected_hash = expected_hash
        self.zipped = zipped
        self.version = version


FANTOM = DownloadableFile(
    url='https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz',
    filename='fantom.tar.gz',
    expected_hash='1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4',
    version="1.0",
    zipped=True
)

# =============================================================================================================

def unzip_file(file_path, directory='.'):
    if file_path.endswith(".zip"):
        target_location =  os.path.join(directory, os.path.splitext(os.path.basename(file_path))[0])
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_location)
    elif file_path.endswith(".tar.gz"):
        target_location =  os.path.join(directory, os.path.basename(file_path).split(".")[0])
        with tarfile.open(file_path) as tar:
            tar.extractall(target_location)

    return target_location

def check_built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version is regarded as not built.
    """
    fname = os.path.join(path, '.built')
    if not os.path.isfile(fname):
        return False
    else:
        with open(fname, 'r') as read:
            text = read.read().split('\n')
        return len(text) > 1 and text[1] == version_string

def mark_built(path, version_string="1.0"):
    """
    Mark this path as prebuilt.
    Marks the path as done by adding a '.built' file with the current timestamp plus a version description string.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)

def download_and_check_hash(url, filename, expected_hash, version, directory='dataset/benchmarks', chunk_size=1024*1024*10):

    # Download the file
    response = requests.get(url, stream=True)
    try:
        total_size = int(response.headers.get('content-length', 0))
    except:
        print("Couldn't get content-length from response headers, using chunk_size instead")
        total_size = chunk_size
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    data = b''
    for chunk in response.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        data += chunk
    progress_bar.close()

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the file to disk
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        f.write(data)

    # Calculate the hash of the downloaded data
    sha256_hash = hashlib.sha256(data).hexdigest()

    # Compare the calculated hash to the expected hash
    if sha256_hash != expected_hash:
        print('@@@ Downloaded file hash does not match expected hash!')
        raise RuntimeError

    return file_path

def build_data(resource, directory='dataset/benchmarks'):
    # check whether the file already exists
    if resource.filename.endswith('.tar.gz'):
        resource_dir = os.path.splitext(os.path.splitext(os.path.basename(resource.filename))[0])[0]
    else:
        resource_dir = os.path.splitext(os.path.basename(resource.filename))[0]
    file_path = os.path.join(directory, resource_dir)

    built = check_built(file_path, resource.version)

    if not built:
        # Download the file
        file_path = download_and_check_hash(resource.url, resource.filename, resource.expected_hash, resource.version, directory)

        # Unzip the file
        if resource.zipped:
            built_location = unzip_file(file_path, directory)
            # Delete the zip file
            os.remove(file_path)
        else:
            built_location = file_path

        mark_built(built_location, resource.version)
        print("Successfully built dataset at {}".format(built_location))
    else:
        print("Already built at {}. version {}".format(file_path, resource.version))
        built_location = file_path

    return built_location

# Function to extract characters from a story
def extract_characters(story):
    lines = story.split("\n")
    characters = {line.split(":")[0] for line in lines if ":" in line}
    return sorted(characters)

class BenchmarkFanToM(BenchmarkBaseline):
    def __init__(self, 
                 is_full: Optional[bool] = True,
                 subset: Optional[int]=None):
        """ 
        """
        super().__init__()
        if is_full:
            self.story = "full_context"
        else:
            self.story = "short_context"

        df = self.load()
        self.data = self.classic_prompting(df, subset)
        self.expanded_df = self.build_expanded_df()
    
    def load(self):
        dpath = build_data(FANTOM)
        file = os.path.join(dpath, "fantom_v1.json")
        df = pd.read_json(file)
        return df
    
    def classic_prompting(self,
                          df: pd.DataFrame,
                          subset: Optional[int]=None):
        # Extract lists of each key into separate columns
        df["question"] = df["beliefQAs"].apply(lambda x: x[0]["question"])
        df["order"] = df["beliefQAs"].apply(lambda x: x[0]["tom_type"])
        df["question_type"] = df["beliefQAs"].apply(lambda x: x[0]["question_type"])
        df["answer"] = df["beliefQAs"].apply(lambda x: x[0]["correct_answer"])
        df["wrong_answer"] = df["beliefQAs"].apply(lambda x: x[0]["wrong_answer"])
        df["cands"] = df.apply(lambda row: [row["answer"] , row["wrong_answer"]], axis=1)
        # Apply the function to each story and create a new column
        df["characters"] = df["full_context"].apply(extract_characters)


        context_template  = (
            "The following multiple choice question is based on the following conversational narrative with the following characters: {characters}. The question "
            "is related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer "
            "from the options provided by printing it as is without any modifications."
        )

        def format_characters(characters_list):
            if len(characters_list) > 1:
                return ", ".join(characters_list[:-1]) + " and " + characters_list[-1]
            return characters_list[0]

        df["prompt"] = df.apply(
            lambda row: (
                f"{context_template.format(characters=format_characters(row['characters']))}\nStory:\n{row[self.story]}\nQuestion: {row['question']}\nOptions:\n" +
                "\n".join([f"- {cand}" for cand in row["cands"]]) +
                "\nAnswer:"
            ),
            axis=1
        )

        # Keep only the False Belief questions
        df = df[df["question_type"] == 'tom:belief:inaccessible'].copy()

        if (subset is not None) and isinstance(subset, int) and (subset > 0) and (subset < len(df)):
            df = df.iloc[:subset]
        
        return df