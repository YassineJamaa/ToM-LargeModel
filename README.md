# The ```ToMechanisms``` repo

[![pages](https://img.shields.io/badge/api-docs-blue)](https://YOUR_GITHUB_NICKNAME.github.io/YOUR_PACKAGE_NAME)

```ToMechanisms``` is an open-source repo that explore mechanistic interpretability of Language and Vision-Language Models to identify networks associated with Theory of Mind task. Inspired by a contrast-based localizer method used in fMRI to identify causal brain region for a specific task, this project builds upon the neuroscientific approach introduced in *`AlKhamissi et al., 2024`*. Their results demonstrate that this method for identifying the brain’s language network can also be effectively applied to LLMs. The purpose of this project is to extend the analysis to Theory of Mind.

In ```ToMechanims```, we propose a framework that enables anyone to apply the functional localizer approach to any LLM or VLM model available on Hugging Face, regardless of model size or the computational resources available, whether using a single GPU, multiple GPUs, or no GPU at all.

![Scheme](assets/scheme_project.png)


**Caution:**: this package is still under development and may change rapidly over the next few weeks.

## Table of Contents

0. Setup


## Setup
1. Create conda environment: `conda create -n llm-loc`
2. Activate environment: `conda activate llm-loc`
3. Install packages: `pip install -r requirements.txt`

## Project Overview

The project consists of implementing an adapted contrast-based analysis method, inspired by task-based fMRI, to identify causal task-relevant neural units in large language models and vision-language models. By drawing parallels to the neuroscientific approach of isolating brain regions responsible for specific task, we apply this methodology to locate and study units critical for Theory of Mind (ToM) tasks.

## To do:
- [ ] Abstract
- [ ] Argument description
