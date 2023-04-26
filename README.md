# Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling

This repository contains the PyTorch implementation for the paper [Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling](https://arxiv.org/abs/2210.12156).
This work has been accepted at the [International Conference on Machine Learning](https://icml.cc/), 2023. 

## 1. Set up environment

### Environment
Run the following commands to create a conda environment:
```bash
conda create -n MulEHR python=3.8
source activate MulEHR
pip install -r requirements.txt
```

### Data 
We uilize open-source EHR [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) to conduct experiment. This dataset is a restricted-access resource. To access the files, you must be a credentialed user and sign the data use agreement (DUA) for the project. Because of the DUA, we cannot provide the data directly.

You need to 
1. Download the MIMIC-III data.
2. Process time serise data following [MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). Note, there are five tasks in the [MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks): in-hospital-mortality, decompensation, length-of-stay, phenotyping and multitask. We conduct experiments on in-hospital-mortality and phenotyping, which are more important based on clinicans' suggestion. Effetiveness of our model on other tasks are leveage as furture works.
3. 


### generation 


### using your own data
