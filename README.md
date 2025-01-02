# The ```ToMechanisms``` repo

[![pages](https://img.shields.io/badge/api-docs-blue)](https://YOUR_GITHUB_NICKNAME.github.io/YOUR_PACKAGE_NAME)

```ToMechanisms``` is an open-source repo that explore mechanistic interpretability of Language and Vision-Language Models to identify networks associated with Theory of Mind task. Inspired by a contrast-based localizer method used in fMRI to identify causal brain region for a specific task, this project builds upon the neuroscientific approach introduced in *`AlKhamissi et al., 2024`*. Their results demonstrate that this method for identifying the brain’s language network can also be effectively applied to LLMs. The purpose of this project is to extend the analysis to Theory of Mind.

In ```ToMechanims```, we propose a framework that enables anyone to apply the functional localizer approach to any LLM or VLM model available on Hugging Face, regardless of model size or the computational resources available, whether using a single GPU, multiple GPUs, or no GPU at all.

![Scheme](assets/scheme_project.png)


**Caution:**: this package is still under development and may change rapidly over the next few weeks.

## Table of Contents

1. [Setup](#setup)   
   - [Dependencies](#dependencies)  

2. [Usage](#usage)  
   - [Quick Start](#quick-start)  
   - [Detailed Examples](#detailed-examples)  

6. [Results and Benchmarks](#results-and-benchmarks)  

7. [Future Work](#future-work)


## Setup

### Dependencies

Firstly, create an enviornment to install the packages
```
conda create -n llm-loc # Create conda environment
conda activate llm-loc # Activate environment
pip install -r requirements.txt # Install packages
```
then, create `.env` file  to store your token from huggingface and the cache directory where Foundation Models will be saved.   
```
# .env file
CACHE_DIR=to/your/directory/cache
HF_ACCESS_TOKEN=<Huggingface token>
 ```

## Usage
### Quick Start



## To do:
- [ ] Abstract
- [ ] Argument description
