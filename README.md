# Inverse-Meta
Optimizing Sampling Patterns for Compressed Sensing MRI with Diffusion Generative Models

## Setup
First, set up a Conda environment using ```conda env create -f conda_env.yml'''.

Download the model checkpoints and fastMRI metadata from: 

## Structure
- **algorithms**: algorithms for solving inverse problems
- **configs**: yaml config files for running experiments
- **datasets**: PyTorch dataset classes
- **metalearners**: the main control classes for gradient-based meta-learning
- **problems**: defines forward operators as classes for re-usability
- **utils_new**: useful functions for experiment logging, metrics, and losses
- ```main.py```: program to invoke for running meta-learning from command line

## Settings


## How to run


## Submodule initialization
```
git submodule update --init --recursive
```

## Download checkpoints

