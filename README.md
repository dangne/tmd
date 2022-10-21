# Textual Manifold-based Defense Against Natural Language Adversarial Examples

This is the official GitHub repository for the following paper:

> [**Textual Manifold-based Defense Against Natural Language Adversarial Examples.**]()  
> Dang Minh Nguyen and Anh Tuan Luu.  
> _Empirical Methods in Natural Language Processing (EMNLP)_, 2022. 



## Installation

To install the required dependencies for this project, run the following commands:

```bash
conda env create -f environment.yml
conda activate tmd
```



## Training

There are two main training steps in our paper:

1. Fine-tuning language models (LMs) on downstream tasks (optional)
2. Training InfoGAN to approximate the natural embedding manifolds of the target LMs

We use [wandb sweep](https://docs.wandb.ai/guides/sweeps) to monitor both the fine-tuning of LMs and the training of InfoGANs. The YAML configuration files for these processes can be found in `./src/finetune` and `./src/train`, respectively, and can be run using the following wandb command:

```bash
wandb sweep <yaml_file>.yaml
```



## Evaluation

To evaluate the robustness of our defenses against different attacks, we follow the recommendations of [Li et al. (2021)](https://aclanthology.org/2021.emnlp-main.251.pdf). Their code can be found [here](https://github.com/RockyLzy/TextDefender).
