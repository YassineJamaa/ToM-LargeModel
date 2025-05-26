# The ```ToMechanisms``` repo

[![pages](https://img.shields.io/badge/api-docs-blue)](https://YOUR_GITHUB_NICKNAME.github.io/YOUR_PACKAGE_NAME)

```ToMechanisms``` is an open-source repo that explore mechanistic interpretability of Language and Vision-Language Models to identify networks associated with Theory of Mind task. Inspired by a contrast-based localizer method used in fMRI to identify causal brain region for a specific task, this project builds upon the neuroscientific approach introduced in *`AlKhamissi et al., 2024`*. Their results demonstrate that this method for identifying the brain’s language network can also be effectively applied to LLMs. The purpose of this project is to extend the analysis to Theory of Mind.

In ```ToMechanims```, we propose a framework that enables anyone to apply the functional localizer approach to any LLM or VLM model available on Hugging Face, regardless of model size or the computational resources available, whether using a single GPU, multiple GPUs, or no GPU at all.

![Scheme](assets/scheme_project.png)


# Abstract

We investigate the use of contrast-based functional localizers–a paradigm inspired by cognitive neuroscience–to identify key neural units that underpin social and mathematical reasoning in large foundation models. Motivated by recent breakthroughs in large language models and advances in mechanistic interpretability, our study adopts the traditional approaches to capture model correlates of Theory-of-Mind (ToM) and Multiple-Demand (MD) processes. Using curated task contrasts, we localized candidate units across a range of transformer-based language models and vision-language models, and then applied targeted lesioning to assess their causal role in task performance on false-belief tasks and math reasoning. Two contrast localizer methods were evaluated: one in which the t-distribution was computed after converting the activation units into their absolute values, and an alternate method that computed the t-distribution directly on the raw, signed activations. For both approaches, while lesioning the most activated units does not induce a significant degradation in accuracy beyond that observed with random unit ablations, unexpected behaviors were observed: lesioning the least activated units sometimes led to either performance degradation or improvement, and the MD localizer impaired performance on both math and ToM tasks. These findings raise concerns about the specificity of the current localizer design for isolating MD and ToM reasoning, highlighting the need for further investigation to refine the method and more accurately capture task-specific activation dynamics.

## Table of Contents

1. [Setup](#setup)   
   - [Dependencies](#dependencies)  

2. [Usage](#usage)  
   - [Quick Start](#quick-start)  
   - [Detailed Examples](#detailed-examples)  


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




