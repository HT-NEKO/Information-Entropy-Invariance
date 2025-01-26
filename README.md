# Information Entropy Invariance: Enhancing Length Extrapolation in Attention Mechanisms

This repo provides the official implementation of our paper "Information Entropy Invariance: Enhancing Length Extrapolation in Attention Mechanisms"

## Summary
In this paper, the authors propose new scaled temperatures based on entropy invariance, **CosScale** and **InfoScale**, to enhance length extrapolation performance, achieving extrapolation up to **64** times the training length.

## Abstract
 Since the emergence of research on improving the length extrapolation capabilities of large language models in 2021, some studies have made modifications to the scaling factor in the scaled dot-product attention mechanism as part of their proposed methods without rigorous theoretical justifications. To fill this gap, we propose two new scaled temperatures based on information entropy invariance to enhance length extrapolation. First, a training-free method InfoScale is designed for dotproduct attention, and preserves focus on original tokens during length extrapolation by ensuring consistent entropy. Second, we theoretically analyze the impact of scaling (CosScale) on cosine attention. Experimental data demonstrates that combining InfoScale and CosScale achieves state-ofthe-art performance on the GAU-α model with a context window extended to 64 times the training length, and outperforms seven existing methods. Our analysis reveals that significantly increasing CosScale approximates the Windowed Attention, and highlights the significance of attention score dilution as a key challenge in long-range context handling.

## Checkpoints
| Model | Starting Point | Train Length | HF Repo |
| --- | --- | --- | --- |
| GAU-alpha_noCosScale | GAU-alpha | 64 | [link](https://huggingface.co/HT-NEKO/GAU-alpha_noCosScale) |
| GAU-alpha_CosScale | GAU-alpha | 64 | [link](https://huggingface.co/HT-NEKO/GAU-alpha_CosScale) |


## Directory Structure
```
├── README.md
├── environment.yml
└── model
    ├── my_utils.py                 # utils
    ├── overlap_trainer.py          # rewrite part of transformers
    ├── train.py                    # training purposes
    ├── train.sh                    # training purposes
    ├── evaluation.py               # evaluation purposes
    ├── evaluation.sh               # evaluation purposes
    ├── my_gau_alpha_eval
    │   ├── layer.py                # GAU-alpha
    │   └── modeling_gau_alpha.py   # GAU-alpha
    └── models_WanJuan
        └── ...                     # models
```

## Usage

### Environment Setup

```
conda env create -f environment.yml
```

## Train
run ```bash train.sh```

## Evaluation
run ```bash evaluation.sh```

## Acknowledgement
We sincerely appreciate the contributions of the following open-source initiatives, which have greatly supported the development of our work:
- [GAU-alpha](https://github.com/ZhuiyiTechnology/GAU-alpha): Transformer model based on Gated Attention Unit.
- [GAU-alpha-pytorch](https://github.com/JunnYu/GAU-alpha-pytorch): The pytorch version of [GAU-alpha](https://github.com/ZhuiyiTechnology/GAU-alpha)

