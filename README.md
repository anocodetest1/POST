# POST: Prior-Observation Adversarial Learning of Spatio–Temporal Associations for Multivariate Time Series Anomaly Detection

This repository contains the official PyTorch implementation of the paper **POST: Prior-Observation Adversarial Learning of Spatio-Temporal Associations for Multivariate Time Series Anomaly Detection**.

## 💡 Highlights
 - We propose POST, a novel MTSAD model that unifies spatial and temporal anomaly modeling under a joint prior–observation adversarial learning framework. 
 - We construct SMD+, a dataset featuring precise channel-wise annotations. The dataset can serve as a generic testbed for anomaly localization in MTS. 
 - Extensive experiments demonstrate that POST consistently outperforms state-of-the-art (SOTA) baselines in standard anomaly detection tasks. Moreover, the optimization of spatial association discrepancy yields superior anomaly localization, significantly enhancing the diagnostic interpretability of final results.

## Suggested citation

Please consider citing our work:

```
@misc{dycl,
      title={Dynamic Contrastive Learning for Hierarchical Retrieval: A Case Study of Distance-Aware Cross-View Geo-Localization}, 
      author={Suofei Zhang and Xinxin Wang and Xiaofu Wu and Quan Zhou and Haifeng Hu},
      year={2025},
      eprint={2506.23077},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23077}, 
}

```

## Get Started

1. Prerequisites
 - Python >= 3.6
 - PyTorch >= 1.4.0
2. Data Preparation
   You can obtain the four standard benchmarks (SMD, MSL, SMAP, PSM) from this [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
   For our newly proposed SMD+ dataset, please download it from Hugging Face: [SMD+ Dataset on Hugging Face](TODO: Insert Hugging Face Dataset Link).
   Place all downloaded data into the ./dataset/ directory.
3. Pre-trained Models
   The best-performing models on all benchmarks (SMD, MSL, SMAP, SWaT, PSM, and SMD+) are available at: [Pre-trained Checkpoints](TODO: Insert Hugging Face Dataset Link).
   Download and extract the .tar.gz file, and place the pre-trained weights into the ./checkpoints/ folder. Ensure the files are named following the format: <dataset_name>_checkpoint.pth (e.g., SMD_checkpoint.pth).

## Evaluation

To evaluate the trained models on the benchmarks, run the following scripts.

(Note: To evaluate on the SMD+ dataset, we directly use the model trained on the standard SMD dataset. Simply rename or copy the SMD_checkpoint.pth file to ./checkpoints/SMD+_checkpoint.pth before running the evaluation).
```bash
# SMD
python main.py --anomaly_ratio 0.5 --batch_size 64 --mode test --dataset SMD --data_path dataset/SMD --input_c 38 --output_c 38
# MSL
python main.py --anomaly_ratio 1. --batch_size 64 --mode test --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55
# SMAP
python main.py --anomaly_ratio 1. --batch_size 64 --mode test --dataset SMAP --data_path dataset/SMAP --input_c 25 --output_c 25
# SWaT
python main.py --anomaly_ratio 1. --win_size 150 --batch_size 64 --mode test --dataset SWaT --data_path dataset/SWaT --input_c 51 --output_c 51
# PSM
python main.py --anomaly_ratio 1. --batch_size 64 --mode test --dataset PSM --data_path dataset/PSM --input_c 25 --output_c 25
# SMD+
python main_channel.py --anomaly_ratio 0.5 --batch_size 64 --mode test --dataset SMD+ --data_path dataset/SMD+ --input_c 38 --output_c 38
```

## Training

To train the POST model from scratch on the respective datasets, execute the following commands:
```bash
# SMD
python main.py --anomaly_ratio 0.5 --num_epochs 10 --batch_size 64 --mode train --dataset SMD --data_path dataset/SMD --input_c 38 --output_c 38
# MSL
python main.py --anomaly_ratio 1. --num_epochs 10 --batch_size 64 --mode train --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55
# SMAP
python main.py --anomaly_ratio 1. --num_epochs 10 --batch_size 64 --mode train --dataset SMAP --data_path dataset/SMAP --input_c 25 --output_c 25
# SWaT
python main.py --anomaly_ratio 1. --win_size 150 --num_epochs 10 --batch_size 64 --mode train --dataset SWaT --data_path dataset/SWaT --input_c 51 --output_c 51
# PSM
python main.py --anomaly_ratio 1. --num_epochs 10 --batch_size 64 --mode train --dataset PSM --data_path dataset/PSM --input_c 25 --output_c 25
```

## Acknowledgments

We deeply appreciate the authors of the following repository for their brilliant open-source contributions, which provided useful modules and data preprocessing pipelines for our code:

- Anomaly-Transformer: https://github.com/thuml/Anomaly-Transformer

