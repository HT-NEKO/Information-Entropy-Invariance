# Entropy-Invariant Attention Scaling for Long-Context Extrapolation in Language and Genomic Sequence Models

This repo provides the official implementation of our paper "Entropy-Invariant Attention Scaling for Long-Context Extrapolation in Language and Genomic Sequence Models"


## Summary
In this paper, the authors propose new scaled temperatures based on entropy invariance, **CosScale** and **InfoScale**, to enhance length extrapolation performance, achieving extrapolation up to **64** times the training length.


## Abstract
Length extrapolation remains a fundamental challenge for large language models (LLMs), particularly in the context of masked language modeling (MLM) over long sequences—a capability critical for modeling biological sequences. In this work, we address this problem by analyzing attention dispersion and propose **InfoScale**, a training-free entropy-invariant scaling strategy for stabilizing dot-product attention across extended contexts. We further introduce **CosScale**, a complementary entropy-preserving method for cosine-based attention mechanisms. Applied to the RoPE-based GAU-α model, our methods enable extrapolation up to 64× the training length, consistently outperforming seven strong baselines. To evaluate cross-domain generalization, we apply **InfoScale** and **CosScale** to DNABERT2 on long-range genomic tasks, including enhancer-gene interaction prediction and non-coding variant classification. Results show that **InfoScale** offers robust extrapolation across domains, while **CosScale** provides additional gains when integrated with RoPE-based attention. Our methods bridge the gap between language and biological sequence modeling under extreme context lengths.


## Dataset
[HF Repo](https://huggingface.co/datasets/HT-NEKO/WanJuan-Dataset_for_Information-Entropy-Invariance)

We preprocess and partition the [WanJuan Patent dataset](https://huggingface.co/datasets/yuyijiong/LongData-Corpus), with specific details regarding the data splits provided in the paper and accompanying code repository.


## Checkpoints
| Model | Starting Point | Train Length | HF Repo |
| --- | --- | --- | --- |
| GAU-alpha_noCosScale | GAU-alpha | 64 | [link](https://huggingface.co/HT-NEKO/GAU-alpha_noCosScale) |
| GAU-alpha_CosScale | GAU-alpha | 64 | [link](https://huggingface.co/HT-NEKO/GAU-alpha_CosScale) |


## Directory Structure
```
├── README.md
├── environment.yml
├── model
│   ├── my_utils.py                     # utils
│   ├── overlap_trainer.py              # rewrite part of transformers
│   ├── train.py                        # training purposes
│   ├── train.sh                        # training purposes
│   ├── evaluation.py                   # evaluation purposes
│   ├── evaluation.sh                   # evaluation purposes
│   ├── my_gau_alpha_eval
│   │   ├── layer.py                    # GAU-alpha
│   │   └── modeling_gau_alpha.py       # GAU-alpha
│   └── models_WanJuan
│       └── ...                         # models
├── datasets
│   ├── WanJuan_deal.json               # train+test+eval WanJuan datasets after preprocess
│   └── WanJuan_deal_validation.json    # test+eval WanJuan datasets after preprocess
└──task
    ├── DNABERT_2                       # DNABERT_2
    │   └── ...
    ├── LRB                             # downstream task datasets
    │   └── ...
    ├── convert_fna2csv.py              # Construct Dataset
    ├── split_data.py
    └── convert_ncbi2ucsc.py
```


## GAU-alpha Experiments

### Environment Setup
```
conda env create -f environment.yml
```

### Train
run 
```
bash train.sh
```

### Evaluation
run 
```
bash evaluation.sh
```



##  DNA Task
### Environment Setup
```
cd task/DNABERT_2
python3 -m pip install -r requirements.txt
```

### Construct Dataset 
run 
```
bash python .\convert_fna2csv.py --assembly_report_path "./path_to_your_report/GCA_000001405.15_GRCh38_assembly_report.txt" --genome_fasta "./path_to_your_fasta/GCA_000001405.15_GRCh38_genomic.fna" --variant_csv ./path_to_your_data.csv --output_dir ./path_to_your_output_folder --window_size 5000
```

### fine-tuneing
run 
```
train_*.sh
```

### evaluation
run 
```
eval_*.sh
```


## Acknowledgement
We sincerely appreciate the contributions of the following open-source initiatives, which have greatly supported the development of our work:
- [GAU-alpha](https://github.com/ZhuiyiTechnology/GAU-alpha): Transformer model based on Gated Attention Unit.
- [GAU-alpha-pytorch](https://github.com/JunnYu/GAU-alpha-pytorch): The pytorch version of [GAU-alpha](https://github.com/ZhuiyiTechnology/GAU-alpha).
- [LongData-Corpus](https://huggingface.co/datasets/yuyijiong/LongData-Corpus): This dataset contains samples with the length greater than 16k, which can be used for pretraining models with extremely long context lengths.
- [DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2): A foundation model trained on large-scale multi-species genome.
- [Casual eQTL and OMIM Task](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark): genomics long-range benchmark.
- [DNA Sequences](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/): Sequence data is provided for all single organism genome assemblies that are included in NCBI's Assembly resource.
